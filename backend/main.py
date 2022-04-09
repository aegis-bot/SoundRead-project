from flask import Flask, request, jsonify, send_from_directory
import torch
import os
import predictor
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from melody.predictor import EffNetPredictor

import mido

app = Flask(__name__, static_folder='static')

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

        playableMidiPath = "static/temp.mid"
        convert_to_midi(melody_preds, playableMidiPath)                                         

        os.remove(temp_path)

        resp = jsonify({"lyrics": lyric_preds,
                        "melody": playableMidiPath})
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp



def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1]-0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid

def convert_to_midi(predicted_result, output_path):
    mid = notes2mid(predicted_result)
    mid.save(output_path)
    return output_path



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
