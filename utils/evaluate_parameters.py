import nltk
import whisper
import nltk
import logging
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


logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for even more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_fluency(text):
    try:
        words = word_tokenize(text)
        if not words:
            raise ValueError("No words found for fluency validation.")

        word_count = len(words)
        sentence_count = max(1, len(nltk.sent_tokenize(text)))
        avg_words = word_count / sentence_count

        fluency_score = min(10.0, avg_words / 2)
        logging.info(f"Fluency score calculated successfully: {fluency_score}")
        return round(fluency_score, 2)

    except Exception as e:
        logging.exception(f"Error in validate_fluency: {e}")
        return 0.0


def validate_vocabulary(text):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid or empty text input for vocabulary validation.")
        
        words = word_tokenize(text)
        if not words:
            raise ValueError("No words found for vocabulary validation.")

        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        vocab_score = min(10.0, diversity * 10)

        logging.info(f"Vocabulary score calculated successfully: {vocab_score}")
        return round(vocab_score, 2)

    except Exception as e:
        logging.exception(f"Error in validate_vocabulary: {e}")
        return 0.0


def validate_grammar(text):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid or empty text input for grammar validation.")
        
        matches = tool.check(text)
        total_words = max(1, len(text.split()))
        error_rate = len(matches) / total_words
        grammar_score = max(0.0, 10.0 - (error_rate * 50))  # Penalty for more errors

        logging.info(f"Grammar score calculated successfully: {grammar_score}")
        return round(grammar_score, 2)

    except Exception as e:
        logging.exception(f"Error in validate_grammar: {e}")
        return 0.0


def validate_relevance(transcription, topic):
    try:
        if not transcription or not isinstance(transcription, str):
            raise ValueError("Invalid or empty transcription for relevance validation.")
        
        if not topic or not isinstance(topic, str):
            raise ValueError("Invalid or empty topic for relevance validation.")
        
        emb1 = semantic_model.encode(transcription, convert_to_tensor=True)
        emb2 = semantic_model.encode(topic, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()

        relevance_score = round(similarity * 10, 2)

        logging.info(f"Relevance score calculated successfully: {relevance_score}")
        return relevance_score

    except Exception as e:
        logging.exception(f"Error in validate_relevance: {e}")
        return 0.0