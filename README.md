# 🎙️ AI Speech Quality Evaluator

This project is a local, open-source solution that evaluates spoken English quality from an audio file based on four parameters:

- **Fluency** – Speed and sentence structure
- **Vocabulary** – Richness and diversity of words
- **Grammar** – Grammatical correctness
- **Relevance** – Semantic alignment to a given topic

---

## 🚀 Features

✅ Transcribes audio using **OpenAI Whisper (local)**  
✅ Computes fluency, grammar, vocabulary richness, and topic relevance  
✅ Returns structured **JSON scores** for easy integration  
✅ 100% local execution – no API keys required!

---

## 📦 Requirements

Install all required dependencies:

```bash
pip install -r requirements.txt


---

## Download NLTK models

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

---

## How to Run

Place your .wav file in the project folder.

Update the file name and topic in analyze_audio.py.

Run the script:

python analyze_audio.py