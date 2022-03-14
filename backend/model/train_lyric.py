import argparse
import os
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader, random_split
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC

from lyric_data import LyricDataset

MODEL_NAME = "facebook/s2t-small-librispeech-asr"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ENABLE_PROGRESS = True

MAX_SECS = 12

LR = 5e-5
N_EPOCHS = 10
SAVE_DIR = "save/"

TRAIN_BATCH_SIZE = 24
TRAIN_SPLIT = .9
VALID_BATCH_SIZE = 48


def train_lyric(
        path: str,
        model_name: str = MODEL_NAME,
        device: torch.device = DEVICE,
        enable_progress: bool = ENABLE_PROGRESS,
        lr: float = LR,
        max_secs: float = MAX_SECS,
        n_epochs: int = N_EPOCHS,
        save_dir: str = SAVE_DIR,
        train_batch_size: int = TRAIN_BATCH_SIZE,
        train_split: float = TRAIN_SPLIT,
        valid_batch_size: int = VALID_BATCH_SIZE
):
    if "wav2vec" in model_name:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        model.config.ctc_zero_infinity = True
        model.config.ctc_loss_reduction = "mean"
    else:
        processor = Speech2TextProcessor.from_pretrained(model_name)
        model = Speech2TextForConditionalGeneration.from_pretrained(model_name).to(device)

    dataset = LyricDataset(path, processor, max_secs=max_secs)
    n_train = int(len(dataset) * train_split) if train_split < 1. else len(dataset)
    n_valid = len(dataset) - n_train

    train_dataset, valid_dataset = random_split(dataset, [n_train, len(dataset) - n_train]) \
                                       if n_valid > 0 else (dataset, None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=dataset.collate_batch,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        collate_fn=dataset.collate_batch,
        pin_memory=True,
    ) if valid_dataset else None

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    train_losses = []
    valid_losses = []
    best_valid_loss = torch.inf
    best_fn = None

    for epoch in (epoch_iterator := tqdm(range(n_epochs), disable=not enable_progress)):
        model.train()
        for features, labels in (train_iterator := tqdm(train_loader, disable=not enable_progress, leave=False)):
            for key in features:
                features[key] = features[key].to(device)
            for key in labels:
                labels[key] = labels[key].to(device)

            optimizer.zero_grad()
            with amp.autocast():
                loss = model(**features, labels=labels.input_ids).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            train_iterator.set_postfix_str("batch_mean_loss: {:.2f}".format(train_losses[-1]))

        if valid_loader:
            with torch.inference_mode():
                model.eval()
                for features, labels in valid_loader:
                    for key in features:
                        features[key] = features[key].to(device)
                    for key in labels:
                        labels[key] = labels[key].to(device)
                    with amp.autocast():
                        loss = model(**features, labels=labels.input_ids).loss
                    valid_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        valid_loss = sum(valid_losses) / len(valid_losses)
        if valid_loss < best_valid_loss:
            prev_best_fn = best_fn
            ts = time.strftime("%y%m%d%H%M")
            best_fn = os.path.join(save_dir, "{}_{}_{:.2f}.pt".format(model_name.split('/')[-1], ts, valid_loss))
            torch.save(model.state_dict(), best_fn)
            if prev_best_fn:
                try:
                    os.remove(prev_best_fn)
                except OSError:
                    pass
            best_valid_loss = valid_loss
        info = "epoch {}: train_loss:{:.2f} valid_loss:{:.2f}".format(epoch, train_loss, valid_loss)
        epoch_iterator.write(info)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="./data/DSing/DSing_train1")

if __name__ == '__main__':
    args = parser.parse_args()
    train_lyric(args.path)


