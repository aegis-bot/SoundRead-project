import os
import random
from typing import List, Tuple, Union

import torch
import torchaudio
from transformers import BatchEncoding, Speech2TextProcessor, Wav2Vec2Processor


class LyricDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path: str,
            processor: Union[Speech2TextProcessor, Wav2Vec2Processor],
            sample_rate: int = 16_000,
            max_secs: float = 6
    ):
        super(LyricDataset, self).__init__()
        self.recs = {}
        self.lyrics = {}
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_length = self.sample_rate * max_secs
        with open(os.path.join(path, "text")) as f:
            for line in f:
                fn, lyric = line.strip().split(maxsplit=1)
                self.lyrics[fn] = lyric
        no_rec = set()
        for fn in self.lyrics:
            try:
                rec, sample_rate = torchaudio.load(os.path.join(path, "audio", f"{fn}.wav"))
                if sample_rate != self.sample_rate:
                    rec = torchaudio.functional.resample(rec, sample_rate, self.sample_rate)
                self.recs[fn] = rec[0][:self.max_length].numpy()
            except RuntimeError:
                no_rec.add(fn)
        for fn in no_rec:
            del self.lyrics[fn]
        self.fns = list(self.lyrics.keys())

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        fn = self.fns[idx]
        return self.recs[fn], self.lyrics[fn]

    def __len__(self) -> int:
        return len(self.fns)

    def sample(self) -> Tuple[torch.Tensor, str]:
        idx = random.randrange(len(self.fns))
        return self.__getitem__(idx)

    def sample_batch(self, n: int) -> List[Tuple[torch.Tensor, str]]:
        inds = random.choices([i for i in range(self.__len__())], k=n)
        return [self.__getitem__(i) for i in inds]

    def collate_batch(self, batch: List[Tuple[torch.Tensor, str]]) -> Tuple[BatchEncoding, BatchEncoding, List[str]]:
        features, texts = tuple(zip(*batch))
        features = self.processor(
            features,
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        )
        with self.processor.as_target_processor():
            labels = self.processor(
                texts,
                padding=True,
                return_tensors="pt"
            )
            # labels.input_ids.masked_fill_(~labels.attention_mask.bool(), -100)
        return features, labels, texts
