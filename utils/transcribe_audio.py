
import os
import whisper
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for even more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load model
whisper_model = whisper.load_model("base")


def transcribe(file_path):
    try:
        # Check if path is valid
        file_path = Path(file_path).resolve(strict=True)
        logging.info(f"Resolved path --> {file_path}")
        
        # if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at path: {file_path}")

        #file must be an audio file
        if not file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Run transcription
        result = whisper_model.transcribe(str(file_path))
        
        # Check if 'text' key is present in result
        if "text" not in result:
            raise KeyError("No 'text' field found in transcription result.")

        logging.info("Transcription successful.")
        return result["text"]

    except FileNotFoundError as fnf_error:
        logging.error(f"File Error: {fnf_error}")
        return "Error: File not found."
    
    except ValueError as val_error:
        logging.error(f"Format Error: {val_error}")
        return "Error: Unsupported file format."
    
    except KeyError as key_error:
        logging.error(f"Transcription Error: {key_error}")
        return "Error: Unexpected transcription result."

    except Exception as e:
        logging.exception(f"An unexpected error occurred during transcription: {e}")
        return "Error: Failed to transcribe the file."