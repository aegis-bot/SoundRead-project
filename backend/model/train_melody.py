import argparse
import os
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForAudioFrameClassification

from melody_data import MelodyDataset

MODEL_NAME = "facebook/wav2vec2-base"
USE_WEIGHTED_LAYER_SUM = True

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ENABLE_PROGRESS = True

MIN_SECS = 2
MAX_SECS = 6
SAMPLES_PER_TRACK = 2

LR = 5e-5
N_EPOCHS = 1000
SAVE_DIR = "save/"
LOAD = ""#"save/wav2vec2-base_2204100159_e100.pt"

TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 16
VALID_INTERVAL = 20


def train_melody(
        path: str,
        model_name: str = MODEL_NAME,
        use_weighted_layer_sum: bool = USE_WEIGHTED_LAYER_SUM,
        device: torch.device = DEVICE,
        enable_progress: bool = ENABLE_PROGRESS,
        lr: float = LR,
        min_secs: float = MIN_SECS,
        max_secs: float = MAX_SECS,
        samples_per_track: int = SAMPLES_PER_TRACK,
        n_epochs: int = N_EPOCHS,
        save_dir: str = SAVE_DIR,
        load: str = LOAD,
        train_batch_size: int = TRAIN_BATCH_SIZE,
        valid_batch_size: int = VALID_BATCH_SIZE,
        valid_interval: int = VALID_INTERVAL
):

    processor = Wav2Vec2Processor.from_pretrained(model_name)

    train_dataset = MelodyDataset(
        os.path.join(path, "train"), processor,
        sample_rate=16_000, label_sample_rate=50,
        min_secs=min_secs, max_secs=max_secs, samples_per_track=samples_per_track
    )
    print(f"lowest note: {train_dataset.min_note} highest note: {train_dataset.max_note}")
    valid_dataset = MelodyDataset(
        os.path.join(path, "valid"), processor,
        sample_rate=16_000, label_sample_rate=50,
        min_secs=4, max_secs=max_secs, samples_per_track=4
    )

    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        model_name,
        num_labels=train_dataset.max_note-train_dataset.min_note+2,
        use_weighted_layer_sum=use_weighted_layer_sum
    ).to(device)
    if load:
        model.load_state_dict(torch.load(load))


    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=train_dataset.collate_batch,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        collate_fn=valid_dataset.collate_batch,
        pin_memory=True,
        shuffle=True,
    ) if valid_dataset else None

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    best_valid_loss = torch.inf
    best_fn = None
    for epoch in (epoch_iterator := tqdm(range(n_epochs), disable=not enable_progress)):
        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []

        model.train()
        for features, labels in (train_iterator := tqdm(train_loader, disable=not enable_progress, leave=False)):
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                logits = model(**features).logits
            labels = labels[:, :logits.shape[1]]
            loss = nn.functional.cross_entropy(
                logits.view(-1, train_dataset.n_classes),
                labels.flatten()
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            _, preds = logits.max(-1)
            train_accs.append(
                ((preds[labels != -100] == labels[labels != -100]).sum() /
                 labels[labels != -100].numel()).item())
            train_iterator.set_postfix_str(
                "batch_mean_loss:{:.2f}  batch_acc:{:.2f}".format(train_losses[-1], train_accs[-1])
            )

        if valid_loader:
            with torch.inference_mode():
                model.eval()
                for features, labels in valid_loader:
                    for key in features:
                        features[key] = features[key].to(device)
                    labels = labels.to(device)
                    with amp.autocast():
                        logits = model(**features).logits
                    labels = labels[:, :logits.shape[1]]
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, valid_dataset.n_classes),
                        labels.flatten()
                    )
                    valid_losses.append(loss.item())
                    _, preds = logits.max(-1)
                    valid_accs.append(
                        ((preds[labels != -100] == labels[labels != -100]).sum() /
                         labels[labels != -100].numel()).item())

        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        info = '='*120 + "\nepoch {}: train_loss:{:.2f} train_acc:{:.2f}".format(epoch, train_loss, train_acc)
        if (epoch % valid_interval == 0 or epoch == n_epochs-1) and valid_loader:
            valid_loss = sum(valid_losses) / len(valid_losses)
            valid_acc = sum(valid_accs) / len(valid_accs)
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
            info += " | valid_loss:{:.2f} valid_acc:{:.2f}".format(valid_loss, valid_acc)
        epoch_iterator.write(info)


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="./data/MIR-ST500/")

if __name__ == '__main__':
    args = parser.parse_args()
    train_melody(args.path)


