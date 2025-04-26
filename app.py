import streamlit as st
import tempfile 

from utils import transcribe_audio, analyse_audio, evaluate_parameters
from utils import transcribe_audio, analyse_audio, evaluate_parameters


st.title("Complete Audio Analyzer Application")

# Upload audio
audio_path = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

# Topic input
topic = st.text_input("Enter the topic/question:")

# Button
if st.button("Analyze"):
    if audio_path and topic:
        with st.spinner("Analyzing..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                            temp_audio.write(audio_path.read())
                            temp_audio_path = temp_audio.name
        transcription = transcribe_audio.transcribe(temp_audio_path)
        scores = analyse_audio.analyse(transcription, topic)
        st.success("Analysis Complete!")
        st.json(scores)
    else:
        st.warning("Please upload an audio file and enter a topic.")
