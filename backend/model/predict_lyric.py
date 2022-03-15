import argparse

import torch
import torchaudio
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input audio file path")

# download lyric_model.pt from https://drive.google.com/file/d/1cJ0X_UN48ysH8JFXQ4VUUk2a3kWWqTVC/view?usp=sharing
if __name__ == '__main__':
    args = parser.parse_args()

    # load model
    pretrained = "facebook/s2t-medium-librispeech-asr"
    save = "lyric_model.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    processor = Speech2TextProcessor.from_pretrained(pretrained, do_upper_case=True)
    model = Speech2TextForConditionalGeneration.from_pretrained(pretrained).to(device)
    model.load_state_dict(torch.load(save))

    # predict
    rec, sample_rate = torchaudio.load(args.input)
    if sample_rate != 16_000:
        torchaudio.functional.resample(rec, sample_rate, 16_000)
    features = processor(
            rec[0].numpy(),
            return_tensors="pt",
            sampling_rate=16_000,
    )
    for k in features:
        features[k] = features[k].to(device)
    preds = processor.decode(model.generate(**features)[0], skip_special_tokens=True)
    print(preds)





