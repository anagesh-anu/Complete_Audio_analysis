import nltk
import whisper
import nltk
import language_tool_python
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')


semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
#tool = language_tool_python.LanguageTool('en-US')
tool = language_tool_python.LanguageToolPublicAPI('en-US')



def compute_fluency(text):
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = max(1, len(nltk.sent_tokenize(text)))
    avg_words_per_sentence = word_count / sentence_count
    fluency_score = min(10.0, avg_words_per_sentence / 2)  # heuristic
    return round(fluency_score, 2)


def compute_vocabulary(text):
    words = word_tokenize(text)
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    vocab_score = min(10.0, diversity * 10)  # 0.0 to 10.0
    return round(vocab_score, 2)


def compute_grammar(text):
    matches = tool.check(text)
    error_rate = len(matches) / max(1, len(text.split()))
    grammar_score = max(0.0, 10.0 - (error_rate * 50))  # Heuristic penalty
    return round(grammar_score, 2)

def compute_relevance(transcription, topic):
    emb1 = semantic_model.encode(transcription, convert_to_tensor=True)
    emb2 = semantic_model.encode(topic, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round(similarity * 10, 2)

