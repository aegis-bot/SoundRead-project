import torchaudio
import glob
import os
import re
from melody.melody_data import transcribe_file


def predict_lyrics(filepath, model, processor, device):
    """
    Predict the lyrics of the music
    :param filepath: path to load the model
    :param model: the model use for prediction
    :param processor: the processor to decode the string
    :param device: device used for prediction(e.g. cpu or cuda)
    :return:the predicted result
    """
    rec, sample_rate = torchaudio.load(filepath)

    if sample_rate != 16_000:
        torchaudio.functional.resample(rec, sample_rate, 16_000)
    features = processor(rec[0].numpy(), return_tensors="pt", sampling_rate=16_000, )

    for k in features:
        features[k] = features[k].to(device)
    preds = processor.decode(model.generate(**features)[0], skip_special_tokens=True)

    return preds


def predict_melody(melody_processor, melody_model, wave_file_path):
    """
    Predict the melody of the music
    :param melody_processor: the processor to decode sound wave
    :param melody_model: the model use for prediction
    :param wave_file_path: the path of music file to be transcribe
    :return: path to access midi
    """
    static_fol_path = os.path.join(os.getcwd(), "static")
    if not os.path.exists(static_fol_path):
        os.makedirs(static_fol_path)

    list_of_files = os.listdir(static_fol_path)
    num_of_files = len(list_of_files)
    if num_of_files == 0:
        filenum = 0
    else:
        file_type = r'/*mid'
        files_name = glob.glob(static_fol_path + file_type)
        files_num = [extract_num(file) for file in files_name]
        filenum = max(files_num) + 1

    relative_midi_path = "static/result" + str(filenum) + ".mid"
    absolute_midi_path = os.path.join(os.getcwd(), relative_midi_path)
    transcribe_file(melody_processor, melody_model, wave_file_path, save_path=absolute_midi_path)

    return relative_midi_path


def extract_num(abs_path):
    """
    extract the number out from the filename
    :param abs_path: the absolute path of the files (including file name)
    :return: the number contain in the file name(e.g. 12 for result12.mid)
    """
    return int(re.findall(r'\d+', os.path.basename(abs_path))[0])
