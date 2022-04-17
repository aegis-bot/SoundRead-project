import torchaudio


def predict_lyrics(filepath, model, processor, device):
    rec, sample_rate = torchaudio.load(filepath)

    if sample_rate != 16_000:
        torchaudio.functional.resample(rec, sample_rate, 16_000)
    features = processor(rec[0].numpy(), return_tensors="pt", sampling_rate=16_000, )

    for k in features:
        features[k] = features[k].to(device)
    preds = processor.decode(model.generate(**features)[0], skip_special_tokens=True)

    return preds
