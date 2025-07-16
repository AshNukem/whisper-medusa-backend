from fastapi import FastAPI, UploadFile, File, HTTPException
import openai
import os
import shutil

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
async def root():
    return {"message": "Whisper Medusa Backend running"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio type")
    
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    transcription = openai.Audio.transcribe(
        model="whisper-1",
        file=open(temp_file_path, "rb")
    )
    
    os.remove(temp_file_path)
    return {"transcription": transcription["text"]}
