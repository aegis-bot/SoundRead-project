# SoundRead Project
SoundRead is a melody and lyrics transcription web service that takes advantage of the modern ML to churn out accurate lyrics.
This program uses a web application frontend (Using Angular) and a backend server using FastAPI. To get started, you need to run both the frontend and the backend.
## Frontend
- Make sure you have angular installed. To install in the Terminal, run ``npm install - g @angular/cli``.

## Run Angular Frontend
- Navigate to frontend folder.

- Run ``ng serve``. Open your browser at ``http://localhost:4200/``

## Backend
- Ensure you have a python virtual environment of your choice like `pyenv` or `Anaconda`.

- Activate your virtual environment.

- Navigate to the `backend` folder

- In virtual environment `(venv)`, install dependencies with: `pip install -r requirements.txt`

## Run python backend server

- In virtual environment `(venv)`, to run python backend server: `uvicorn main:app --reload`

Server runs by default at http://localhost:8000

Documentation for api will also be available at http://localhost:8000/docs#/
