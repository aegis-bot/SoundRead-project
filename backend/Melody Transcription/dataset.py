from genericpath import exists
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import concurrent.futures
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


def run_spleeter(data_dir):

    from spleeter.separator import Separator
    import soundfile

    separator = Separator('spleeter:2stems')

    for d in os.listdir(data_dir):
        print("Isolating: " + d)
        audio_path = os.path.join(data_dir, d, "Mixture.mp3")
        vocal_path = os.path.join(data_dir, d, "Vocal.wav")
        if exists(vocal_path):
            print("Isolated vocals exists: " + d)
            continue
        y, sr = librosa.core.load(audio_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

        waveform = np.expand_dims(y, axis=1)
        prediction = separator.separate(waveform, "")  # can be done using Tensorflow or Librosa
        
        vocals = librosa.core.to_mono(prediction["vocals"].T)
        vocals = np.clip(vocals, -1.0, 1.0)

        instrumental = librosa.core.to_mono(prediction["accompaniment"].T)
        instrumental = np.clip(instrumental, -1.0, 1.0)

        soundfile.write(os.path.join(data_dir, d, "Vocal.wav"), vocals, 44100, subtype='PCM_16')
        soundfile.write(os.path.join(data_dir, d, "Instrumental.wav"), instrumental, 44100, subtype='PCM_16')
        print("Written: " + d)


def preprocess(annotation_data, length):

    # annotation format: [onset, offset, note_number]
    new_label = []

    cur_note = 0
    cur_note_onset = annotation_data[cur_note][0]
    cur_note_offset = annotation_data[cur_note][1]
    cur_note_number = annotation_data[cur_note][2]

    # note numbers ranging from C2 (36) to B5 (83)
    # octave class ranging from 0 to 4, octave class 0 to 3: octave 2 to 5, octave class 4: unknown class(silence)
    # pitch_class ranging from 0 to 12, pitch class 0 to 11: pitch C to B, pitch class 12: unknown class(silence)
    note_start = 36
    frame_size = 1024.0 / 44100.0

    is_onset_assigned = False

    # label format: [0/1, 0/1, octave_class_num, pitch_class_num]
    for i in range(length):
        cur_time = i * frame_size
        label = [0, 0, 0, 0]

        # Finding the correct octave and pitch
        # 0: 36-47, 1: 48-59, 2: 60-71, 3: 72-83, 4: unknown octave
        octave = int((cur_note_number - note_start) / 12)
        pitch = int(cur_note_number % 12)
        next_time = (i + 1) * frame_size

        if cur_time <= cur_note_onset < next_time and not is_onset_assigned:  # Onset frame case
            label = [1, 0, octave, pitch]
            is_onset_assigned = True

        elif cur_time <= cur_note_offset < next_time and is_onset_assigned:  # Offset frame case
            label = [0, 1, octave, pitch]
            is_onset_assigned = False
            cur_note += 1  # offset has been assigned, advance to next node
            if cur_note < len(annotation_data):
                cur_note_onset = annotation_data[cur_note][0]
                cur_note_offset = annotation_data[cur_note][1]
                cur_note_number = annotation_data[cur_note][2]

        elif cur_note_onset < cur_time < cur_note_offset:  # Voiced frame case
            if not is_onset_assigned:
                label = [1, 0, octave, pitch]  # to catch any onset frames missed
                is_onset_assigned = True
            else:
                label = [0, 0, octave, pitch]  # normal voiced frame

        elif is_onset_assigned:  # to catch any offset frames missed
            label = [0, 1, octave, pitch]
            is_onset_assigned = False
            cur_note += 1
            if cur_note < len(annotation_data):
                cur_note_onset = annotation_data[cur_note][0]
                cur_note_offset = annotation_data[cur_note][1]
                cur_note_number = annotation_data[cur_note][2]

        else:  # Silent frame case
            label = [0, 0, 4, 12]

        new_label.append(label)

    return np.array(new_label)


def get_feature(y):
    # Computes Constant-Q Transform of the given signals
    y = librosa.util.normalize(y)
    cqt_feature = np.abs(librosa.cqt(y, sr=44100, hop_length=1024, fmin=librosa.midi_to_hz(36), n_bins=84*2, bins_per_octave=12*2, filter_scale=1.0)).T
    return torch.tensor(cqt_feature, dtype=torch.float).unsqueeze(1)


def main(data_dir, gt_path, output_dir, filename_trail):
    print("Generating dataset...")
    print("Using directory: {}".format(data_dir))

    # Write the datasets into binary files
    dataset = AudioDataset(gt_path=gt_path, data_dir=data_dir)
    target_path = Path(output_dir) / (Path(data_dir).stem + filename_trail)
    with open(target_path, 'wb') as f:
        pickle.dump(dataset, f)
    print("Dataset generated at {}".format(target_path))


class AudioDataset(Dataset):

    def __init__(self, gt_path, data_dir=None):

        with open(gt_path) as json_data:
            gt = json.load(json_data)
        
        self.data = []
        self.ans = []
        self.pitch = []

        temp_cqt = {}
        future = {}
        print("Computing CQT...")

        frame_size = 1024.0 / 44100.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for d in os.listdir(data_dir):
                wav_path = os.path.join(data_dir, d, "Vocal.wav")
                y, sr = librosa.core.load(wav_path, sr=None, mono=True)
                if sr != 44100:
                    y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
                future[d] = executor.submit(get_feature, y)
        
        for d in os.listdir(data_dir):
            temp_cqt[d] = future[d].result()
        
        print("Creating dataset...")

        for d in tqdm(os.listdir(data_dir)):
            cqt_data = temp_cqt[d]
            gt_data = gt[d]
            ans_data = preprocess(gt_data, cqt_data.shape[0])

            frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
            padding = torch.zeros((channel_num, cqt_size), dtype=torch.float)

            for frame in range(frame_num):
                cqt_feature = []
                for window in range(frame - 5, frame + 6):
                    if window < 0 or window >= frame_num:
                        cqt_feature.append(padding.unsqueeze(1))
                    else:
                        cqt_feature.append(cqt_data[frame].unsqueeze(1))
                
                cqt_feature = torch.cat(cqt_feature, dim=1)
                self.data.append((cqt_feature, ans_data[frame]))

    
    def __getitem__(self, idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)


def run_spleeter_one_song(y, sr):
    
    from spleeter.separator import Separator
    warnings.filterwarnings('ignore')
    
    separator = Separator('spleeter:2stems')

    if sr != 44100:
        y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)

    waveform = np.expand_dims(y, axis=1)
    prediction = separator.separate(waveform, "")
    
    vocals = librosa.core.to_mono(prediction["vocals"].T)
    vocals = np.clip(vocals, -1.0, 1.0)
    del separator
    return vocals, 44100


class OneSong(Dataset):
    '''
        The Dataset class is used for preprocessing and preparing testing data. 
        The difference is that this class is only used to prepare data of one song with song id and without annotations. 
        Args:
            input_path: the path to one song e.g. "./data/test/100/Mixture.mp3"
            song_id: id of the song e.g. 100 
    '''
    def __init__(self, input_path, song_id):

        y, sr = librosa.core.load(input_path, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
        y = librosa.util.normalize(y)
        y, sr = run_spleeter_one_song(y, sr)
        
        self.data_instances = []
        cqt_data = get_feature(y)
        frame_num, channel_num, cqt_size = cqt_data.shape[0], cqt_data.shape[1], cqt_data.shape[2]
        zeros_padding = torch.zeros((channel_num, cqt_size), dtype=torch.float)

        for frame_idx in range(frame_num):
            cqt_feature = []
            for frame_window_idx in range(frame_idx - 5, frame_idx + 6):
                # padding with zeros if needed
                if frame_window_idx < 0 or frame_window_idx >= frame_num:
                    cqt_feature.append(zeros_padding.unsqueeze(1))
                else:
                    choosed_idx = frame_window_idx
                    cqt_feature.append(cqt_data[choosed_idx].unsqueeze(1))

            cqt_feature = torch.cat(cqt_feature, dim=1)
            self.data_instances.append((cqt_feature, song_id))

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __len__(self):
        return len(self.data_instances)


if __name__ == '__main__':
    """
    This script performs preprocessing raw data and prepares train/valid data.
    
    Sample usage:
    python dataset.py
    """

    train_data_dir = './data/train/'
    valid_data_dir = './data/valid/'
    gt_path = './data/MIR-ST500_20210206/MIR-ST500_corrected.json'
    output_dir = './data'
    filename_trail = '.pkl'
    
    run_spleeter(train_data_dir)
    run_spleeter(valid_data_dir)
    main(train_data_dir, gt_path, output_dir, filename_trail)
    main(valid_data_dir, gt_path, output_dir, filename_trail)
