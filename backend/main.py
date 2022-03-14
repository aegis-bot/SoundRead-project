from email.mime import multipart
from enum import Enum
from importlib.resources import contents
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, Form, UploadFile

#create a FastAPI instance
app = FastAPI()

origins = [
    "http://localhost:4200",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestData(str, Enum):
    packetName = "packetName"
    video = "video"
    audio = "audio"
    imuData = "imuData"



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/simpleMessage/")
async def getDataset(message: str):
    data = "backend replies you!"
    print(message)
    return {"backendMessage" : data}    




#send files
@app.post("/upload/")
async def uploadFile(fileObject:UploadFile = File(...)):
    print("frontend is uploading file!")
    
    contents = await fileObject.read();
    #save_file(FileObject.filename, contents)
    return {"uploaded file: " : fileObject.filename}

    

def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)
    