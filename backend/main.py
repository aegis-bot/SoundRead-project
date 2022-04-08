from flask import Flask, request, jsonify
import torch
import os
import predictor
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from melody.predictor import EffNetPredictor

app = Flask(__name__)

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
        lyric_preds = predictor.predict_lyrics(temp_path, model, processor, device)

        # melody prediction
        song_id = '1'
        results = {}

        melody_preds = predictor.predict_song(my_melody_predictor, temp_path, song_id, results, do_svs=False,
                                                onset_thres=0.4, offset_thres=0.5)

        os.remove(temp_path)

        resp = jsonify({"lyrics": lyric_preds,
                        "melody": melody_preds})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp


if __name__ == "__main__":
    # load lyric
    pretrained = "facebook/s2t-medium-librispeech-asr"
    lyric_model_path = "model/lyric_model.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    processor = Speech2TextProcessor.from_pretrained(pretrained, do_upper_case=True)
    model = Speech2TextForConditionalGeneration.from_pretrained(pretrained).to(device)
    model.load_state_dict(torch.load(lyric_model_path, map_location=torch.device('cpu')))

    # load melody
    melody_model_path = "model/melody_model"
    my_melody_predictor = EffNetPredictor(device=device, model_path=melody_model_path)

    app.run(debug=True)
