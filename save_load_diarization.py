import pickle
import json
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm

def save_diarization_pickle(diarization, filename):
    """
    Saves the diarization object (Annotation) using pickle.
    This preserves the entire Python object.
    """
    with open(filename, "wb") as f:
        pickle.dump(diarization, f)
    print(f"Saved diarization to {filename} (pickle format)")

def load_diarization_pickle(filename):
    """
    Loads a diarization object from a pickle file.
    """
    with open(filename, "rb") as f:
        diarization = pickle.load(f)
    print(f"Loaded diarization from {filename}")
    return diarization

def save_diarization_rttm(diarization, filename):
    """
    Saves the diarization in RTTM format.
    RTTM is the standard format for speaker diarization.
    """
    with open(filename, "w") as f:
        diarization.write_rttm(f)
    print(f"Saved diarization to {filename} (RTTM format)")

def load_diarization_rttm(filename):
    """
    Loads diarization from an RTTM file.
    Returns the first Annotation object found in the file.
    """
    annotations = load_rttm(filename)
    # load_rttm returns a dict: {uri: Annotation}
    if not annotations:
        return None
    diarization = next(iter(annotations.values()))
    print(f"Loaded diarization from {filename}")
    return diarization

def save_diarization_json(diarization, filename):
    """
    Saves the diarization in JSON format (list of segments).
    This is good for web applications or sharing data.
    """
    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "label": label
        })
    with open(filename, "w") as f:
        json.dump(segments, f, indent=4)
    print(f"Saved diarization to {filename} (JSON format)")

def load_diarization_json(filename, uri="audio"):
    """
    Loads diarization from a JSON file and returns an Annotation object.
    """
    with open(filename, "r") as f:
        segments = json.load(f)

    annotation = Annotation(uri=uri)
    for seg in segments:
        annotation[Segment(seg["start"], seg["end"])] = seg["label"]

    print(f"Loaded diarization from {filename}")
    return annotation

if __name__ == "__main__":
    # Example usage:
    # Assuming 'turns' is what you got from diarization.py

    # Create a dummy Annotation for demonstration if needed
    test_annotation = Annotation(uri="test_audio")
    test_annotation[Segment(0, 1)] = "SPEAKER_01"
    test_annotation[Segment(1, 2)] = "SPEAKER_02"

    # Test Pickle
    save_diarization_pickle(test_annotation, "test.pkl")
    loaded_pkl = load_diarization_pickle("test.pkl")
    print(f"Pickle Load Result: {loaded_pkl}")

    # Test RTTM
    save_diarization_rttm(test_annotation, "test.rttm")
    loaded_rttm_obj = load_diarization_rttm("test.rttm")
    print(f"RTTM Load Result: {loaded_rttm_obj}")

    # Test JSON
    save_diarization_json(test_annotation, "test.json")
    loaded_json = load_diarization_json("test.json")
    print(f"JSON Load Result: {loaded_json}")
