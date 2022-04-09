import json
import os
import pickle
import random
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import BatchEncoding, Wav2Vec2Processor


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
        self.min_note = int(min(note for track in labels.values() for _, _, note in track))
        self.max_note = int(max(note for track in labels.values() for _, _, note in track))
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


if __name__ == '__main__':
    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    for partition in ["train", "valid"]:
        path = os.path.join("./data/MIR-ST500/", partition)
        MelodyDataset(path, processor)
