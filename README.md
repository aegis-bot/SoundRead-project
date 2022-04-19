# SoundRead Project
SoundRead is a melody and lyrics transcription web service that takes advantage of the modern ML to churn out accurate lyrics.
This program uses a web application frontend (Using Angular) and a backend server using FastAPI. To get started, you need to run both the frontend and the backend.
## Frontend
- Make sure you have angular installed. To install in the Terminal, navigate to frontend folder and run ``npm install - g @angular/cli``.

## Run Angular Frontend
- Navigate to frontend folder.

- Run ``ng serve``. Open your browser at ``http://localhost:4200/``

## Backend
- Ensure you have a python virtual environment of your choice like `pyenv` or `Anaconda`.

- Activate your virtual environment.

- In virtual environment `(venv)`, install dependencies with: `pip install -r requirements.txt`

- Make sure that ffmpeg is downloaded (scroll to the last section for the download process)

- Proceed to download the [lyric transcription model](https://drive.google.com/file/d/1E-aDN4qNF5Y-ngqjIuHcui2jHRwGLKPj) and [melody transciption model](https://drive.google.com/file/d/19KTUjcNpOtUD8XIiVNBlFeGaDDgEzoMH)

## Run python backend server
- Navigate to the `backend` folder

- In virtual environment `(venv)`, to run python backend server: `python3 main.py --lyric_model LYRIC_MODEL_PATH --melody_model MELODY_MODEL_PATH`

Server runs by default at http://127.0.0.1:5000/

## Download ffmpeg
If you are using Anaconda, install *ffmpeg* by calling
```
conda install -c conda-forge ffmpeg
```

## To get prediction for lyric and melody
```shell
cd backend/lyric; python predict_lyric.py --input FILE_PATH
cd backend/melody; python predict_melody.py --input FILE_PATH
```
The lyric model and melody model PyTorch weights can be downloaded at
```html
https://drive.google.com/file/d/1E-aDN4qNF5Y-ngqjIuHcui2jHRwGLKPj
https://drive.google.com/file/d/19KTUjcNpOtUD8XIiVNBlFeGaDDgEzoMH
```

If you are not using Anaconda:

* Linux (apt-get): `apt-get install ffmpeg`
* Linux (yum): `yum install ffmpeg`
* Mac: `brew install ffmpeg`
* Windows: download ffmpeg binaries from this [website](https://www.gyan.dev/ffmpeg/builds/)
