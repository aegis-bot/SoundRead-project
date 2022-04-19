from flask import Flask, request, jsonify
import torch
import os
import predictor
import argparse
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2ForAudioFrameClassification
from flask_cors import CORS


app = Flask(__name__, static_folder='static')
cors = CORS(app, resources={r"/static/*": {"origins": "*"}})

parser = argparse.ArgumentParser()
parser.add_argument("--lyric_model", required=True, help="input lyric model path")
parser.add_argument("--melody_model", required=True, help="input melody model path")

file_type_allowed = ['mp3', 'wav']


def allowed_file(filename):
    """
    check if the input file is allow to be process
    :param filename: the name of the file submitted
    :return: boolean
    """
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
        midi_path = predictor.predict_melody(melody_processor, melody_model, temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        resp = jsonify({"lyrics": lyric_preds,
                        "melody": midi_path})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load lyric
    pretrained = "facebook/s2t-medium-librispeech-asr"
    lyric_model_path = args.lyric_model
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
    melody_model_path = args.melody_model
    melody_model.load_state_dict(torch.load(melody_model_path, map_location=torch.device('cpu')))

    app.run(debug=True)
