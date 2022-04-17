from flask import Flask, request, jsonify
import torch
import os
import predictor
import glob
import re
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2ForAudioFrameClassification
from flask_cors import CORS

from melody.melody_data import transcribe_file

app = Flask(__name__, static_folder='static')
cors = CORS(app, resources={r"/static/*": {"origins": "*"}})

file_type_allowed = ['mp3', 'wav']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in file_type_allowed


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if request.files['fileObject'].filename == '':
            resp = jsonify({"response": "No file input in the request"})
            resp.status_code = 400
            return resp

        file = request.files['fileObject']

        if not allowed_file(file.filename):
            resp = jsonify({"response": "File type not allowed"})
            return resp

        temp_path = os.path.join(os.getcwd(), file.filename)
        file.save(temp_path)

        # lyric prediction
        lyric_preds = predictor.predict_lyrics(temp_path, lyric_model, lyric_processor, device)

        # melody prediction
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
        transcribe_file(melody_processor, melody_model, temp_path, save_path=absolute_midi_path)

        os.remove(temp_path)

        resp = jsonify({"lyrics": lyric_preds,
                        "melody": relative_midi_path})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp

#get the number from the filename
def extract_num(abs_path):
    return int(re.findall(r'\d+', os.path.basename(abs_path))[0])


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load lyric
    pretrained = "facebook/s2t-medium-librispeech-asr"
    lyric_model_path = "model/lyric_model.pt"
    lyric_processor = Speech2TextProcessor.from_pretrained(pretrained, do_upper_case=True)
    lyric_model = Speech2TextForConditionalGeneration.from_pretrained(pretrained).to(device)
    lyric_model.load_state_dict(torch.load(lyric_model_path, map_location=torch.device('cpu')))

    # load melody
    min_note = 29
    max_note = 83
    melody_model_name = "facebook/wav2vec2-base"
    melody_processor = Wav2Vec2Processor.from_pretrained(melody_model_name)
    melody_model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        melody_model_name,
        use_weighted_layer_sum=True,
        num_labels=max_note - min_note + 2
    ).to(device)
    melody_model_path = "model/melody_model.pt"
    melody_model.load_state_dict(torch.load(melody_model_path, map_location=torch.device('cpu')))

    app.run(debug=True)
