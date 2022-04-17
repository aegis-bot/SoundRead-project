import argparse

import torch
from transformers import *

from melody_data import *

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input audio file path")

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    min_note = 29
    max_note = 83

    model_name = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        model_name,
        use_weighted_layer_sum=True,
        num_labels=max_note - min_note + 2
    ).to(device)

    save = "save/wav2vec2-base_2204100159_e100.pt"

    if save:
        model.load_state_dict(torch.load(save))

    transcribe_file(processor, model, args.input)

