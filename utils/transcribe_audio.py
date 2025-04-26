
import os
import whisper
from pathlib import Path


# Load models
whisper_model = whisper.load_model("base")


def transcribe_audio(file_path):
    file_path = Path(file_path).resolve(strict=True)  # resolve to absolute path
    print("Resolved path -->", file_path)
    print("Exists:", file_path.exists())

    result = whisper_model.transcribe(str(file_path))  # convert Path to string
    return result["text"]