import json
import os
import pickle
import random
from typing import List, Tuple

import librosa
import mido
import numpy as np
from tqdm import tqdm

import torch
from transformers import BatchEncoding, Wav2Vec2Processor, Wav2Vec2ForAudioFrameClassification


class MelodyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str,
            processor: Wav2Vec2Processor,
            sample_rate: int = 16_000,
            label_sample_rate: int = 50,
            min_secs: float = 1,
            max_secs: float = 4,
            samples_per_track: int = 8
    ):
        super(MelodyDataset, self).__init__()
        self.recs = {}
        self.processor = processor
        self.sample_rate, self.label_sample_rate = sample_rate, label_sample_rate
        self.min_secs, self.max_secs = min_secs, max_secs
        self.samples_per_track = samples_per_track

        with open(os.path.join(path, "MIR-ST500_corrected.json")) as f:
            labels = json.load(f)
        self.min_note = int(min(note for track in labels.values() for _, _, note in track))  # 29
        self.max_note = int(max(note for track in labels.values() for _, _, note in track))  # 83
        self.n_classes = self.max_note - self.min_note + 2
        exist_fns = [dn for dn in os.listdir(path) if os.path.isdir(os.path.join(path, dn))]
        self.labels = {fn: label for fn, label in labels.items() if fn in exist_fns}

        pkl_path = os.path.join(path, f"{os.path.basename(path.rstrip('/'))}.pkl")
        try:
            with open(pkl_path, 'rb') as f:
                print(f"loading audio ndarray from {pkl_path}...")
                self.recs = pickle.load(f)
                print("done")
        except FileNotFoundError:
            for fn in tqdm(self.labels):
                rec, sample_rate = librosa.load(os.path.join(path, f"{fn}/", "mixture.mp3"), sr=self.sample_rate)
                self.recs[fn] = rec
            with open(pkl_path, 'wb+') as f:
                print(f"saving audio ndarray to {pkl_path}...")
                pickle.dump(self.recs, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("done")

        self.fns = list(self.recs.keys())

    def __getitem__(self, idx) -> Tuple[np.ndarray, List[List[int]]]:
        fn = self.fns[idx]
        return self.recs[fn], self.labels[fn]

    def __len__(self) -> int:
        return len(self.fns)

    def sample(self) -> Tuple[np.ndarray, List[List[int]]]:
        idx = random.randrange(len(self.fns))
        return self.__getitem__(idx)

    def sample_batch(self, n: int) -> List[Tuple[np.ndarray, List[List[int]]]]:
        inds = random.choices([i for i in range(self.__len__())], k=n)
        return [self.__getitem__(i) for i in inds]

    def collate_batch(self, batch: List[Tuple[np.ndarray, List[List[int]]]]) -> Tuple[BatchEncoding, torch.Tensor]:
        features, labels = [], []
        for feature, label in batch:
            segment_features, segment_labels = self.to_segment_train_samples(feature, label)
            features += segment_features
            labels += segment_labels
        features = self.processor(
            features,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sample_rate
        )
        max_label_len = max(label.shape[0] for label in labels)
        padded_labels = torch.full((len(labels), max_label_len), -100)
        for i, label in enumerate(labels):
            padded_labels[i, :label.shape[0]] = torch.from_numpy(label)

        return features, padded_labels

    def to_segment_train_samples(
            self,
            rec: np.ndarray,
            labels: List[List[int]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n_frames = rec.shape[0]
        train_samples = []
        train_labels = []
        end_frame = 0
        while end_frame < n_frames:
            start_frame = end_frame
            end_frame = (np.random.rand()*(self.max_secs-self.min_secs) + self.min_secs) * self.sample_rate + end_frame
            end_frame = int(min(end_frame, n_frames))
            train_samples.append(rec[start_frame: end_frame])
            start_t = start_frame / self.sample_rate
            end_t = end_frame / self.sample_rate
            sample_labels = np.zeros(((int((end_t - start_t) * self.label_sample_rate)), ), dtype=int)
            for label_start_t, label_end_t, note in labels:
                sample_labels[
                    int((max(label_start_t, start_t) - start_t) * self.label_sample_rate):
                    int(max(min(label_end_t, end_t) - start_t, 0) * self.label_sample_rate)
                ] = min(max(note, self.min_note), self.max_note) - self.min_note + 1

            train_labels.append(sample_labels)
        if len(train_labels) > self.samples_per_track:
            sample_inds = random.sample(range(len(train_labels)), self.samples_per_track)
        else:
            sample_inds = [i for i in range(len(train_labels))]
        return [train_samples[i] for i in sample_inds], [train_labels[i] for i in sample_inds]


def transcribe_file(
        processor: Wav2Vec2Processor,
        model: Wav2Vec2ForAudioFrameClassification,
        path: str,
        note_offset: int = 28,
        label_sample_rate: int = 50,
        window_size: int = 4,
        save_path = "static/result.mid"
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    rec, _ = librosa.load(path, sr=16_000)
    pieces = []
    notes = []
    with torch.inference_mode():
        with torch.no_grad():
            for start in range(-16_000*window_size, rec.shape[0], 16_000*window_size):
                pieces.append(rec[max(0, start): start+16_000*window_size*3])
                if pieces and len(pieces) % 8 == 0 or start+16_000*window_size*2 > rec.shape[0]:
                    features = processor(pieces, padding=True, return_tensors="pt", sampling_rate=16_000)
                    logits = model(features.input_values.to(device)).logits
                    _, preds = logits.max(-1)
                    preds = torch.where(preds > 0, preds + note_offset, 0)
                    notes.append(preds[:, 50*window_size: 50*window_size*2].flatten().cpu().numpy())
                    pieces.clear()
    notes = np.concatenate(notes)[:int(rec.shape[0]*label_sample_rate/16_000)]
    mid = notes2mid(notes, label_sample_rate)
    mid.save(os.path.join(os.path.dirname(path), save_path))


def notes2mid(notes: np.ndarray, label_sample_rate: int = 50) -> mido.MidiFile:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_t = 0
    cur_note = 0
    cur_ticks = 0
    for i, note in enumerate(notes):
        if note != cur_note and (notes[max(i-2, 0): i+2] == note).all():
        # note = np.bincount(notes[max(i-2, 0): i+2]).argmax()
        # if note != cur_note:
            ticks = int(mido.second2tick(cur_t, ticks_per_beat=480, tempo=new_tempo))
            length = ticks - cur_ticks
            cur_ticks = ticks
            if cur_note == 0:
                track.append(mido.Message('note_on', note=note, velocity=100, time=length))
            else:
                if note == 0:
                    track.append(mido.Message('note_off', note=cur_note, velocity=100, time=length))
                else:
                    track.append(mido.Message('note_off', note=cur_note, velocity=100, time=max(0, length - 160)))
                    track.append(mido.Message('note_on', note=note, velocity=100, time=160))
            cur_note = note
        cur_t += 1 / label_sample_rate
    return mid


if __name__ == '__main__':
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    for partition in ["train", "valid"]:
        path = os.path.join("./data/MIR-ST500/", partition)
        MelodyDataset(path, processor)
