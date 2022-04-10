import torchaudio
from melody.seq_dataset import SeqDataset

def predict_lyrics(filepath, model, processor, device):
    rec, sample_rate = torchaudio.load(filepath)

    if sample_rate != 16_000:
        torchaudio.functional.resample(rec, sample_rate, 16_000)
    features = processor(rec[0].numpy(), return_tensors="pt", sampling_rate=16_000, )

    for k in features:
        features[k] = features[k].to(device)
    preds = processor.decode(model.generate(**features)[0], skip_special_tokens=True)

    return preds


def predict_song(predictor, wav_path, song_id, results, do_svs, onset_thres, offset_thres):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test_dataset = SeqDataset(wav_path, song_id, do_svs)

    results = predictor.predict(test_dataset, results=results, onset_thres=onset_thres, offset_thres=offset_thres)

    return results
