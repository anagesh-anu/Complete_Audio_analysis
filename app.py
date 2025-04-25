import streamlit as st
from utils import transcribe_audio, analyze_speech, evaluate_parameters

st.title("Complete Auido Analyzer 🎙️")

# Upload audio
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

# Topic input
topic = st.text_input("Enter the topic/question:")

# Button
if st.button("Analyze"):
    if audio_file and topic:
        with st.spinner("Analyzing..."):
            transcript = transcribe_audio(audio_file)
            results = evaluate_parameters(transcript, topic)
        st.success("Analysis Complete!")
        st.json(results)
    else:
        st.warning("Please upload an audio file and enter a topic.")
