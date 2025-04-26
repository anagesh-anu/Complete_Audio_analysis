
from utils import transcribe_audio, analyse_audio, evaluate_parameters



def analyse_audio(transcript, topic):
    
    print("\nTranscription:\n", transcript)

    return {
        "fluency": evaluate_parameters.compute_fluency(transcript),
        "vocabulary": evaluate_parameters.compute_vocabulary(transcript),
        "grammar": evaluate_parameters.compute_grammar(transcript),
        "relevance": evaluate_parameters.compute_relevance(transcript, topic)
    }