
import logging
from utils import transcribe_audio, analyse_audio, evaluate_parameters

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for even more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def analyse(transcript, topic):
    try:
        if not transcript or not isinstance(transcript, str):
            raise ValueError("Invalid or empty transcript provided.")
        
        if not topic or not isinstance(topic, str):
            raise ValueError("Invalid or empty topic provided.")

        logging.info("Starting analysis on transcript.")
        logging.debug(f"Transcript: {transcript[:100]}...")  # log first 100 characters

        scores = {
            "fluency": evaluate_parameters.validate_fluency(transcript),
            "vocabulary": evaluate_parameters.validate_vocabulary(transcript),
            "grammar": evaluate_parameters.validate_grammar(transcript),
            "relevance": evaluate_parameters.validate_relevance(transcript, topic)
        }

        logging.info("Analysis completed successfully.")
        logging.debug(f"Scores: {scores}")

        return scores

    except ValueError as ve:
        logging.error(f"ValueError during analysis: {ve}")
        return {"error": str(ve)}
    
    except Exception as e:
        logging.exception(f"Unexpected error during analysis: {e}")
        return {"error": "Failed to analyze transcript."}