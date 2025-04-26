from utils import transcribe_audio, analyse_audio, evaluate_parameters



# Upload audio
audio_path = "C:/Users/anage/Downloads/speech1.mp3"

# Topic input
topic = "The importance of renewable energy"

# Example usage
if __name__ == "__main__":
    topic = "The importance of renewable energy"
    transcription = transcribe_audio.transcribe_audio(audio_path)
    scores = analyse_audio.analyse_audio(transcription, topic)
    print("\nAnalysis Result:")
    print(scores)
