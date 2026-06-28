from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import argparse
from save_load_diarization import save_diarization_pickle

# Get the model from https://huggingface.co/pyannote/speaker-diarization-community-1

def speakers(diarization_model="models/speaker-diarization-community-1", sound="audio.wav", min_speakers=2,
             max_speakers=6):
    pipeline = Pipeline.from_pretrained(diarization_model)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    device = torch.device(device)

    # send pipeline to GPU (when available)
    pipeline.to(device)

    # run the pipeline locally on your computer
    with ProgressHook() as hook:
        output = pipeline(sound, hook=hook, min_speakers=min_speakers, max_speakers=max_speakers)

    save_diarization_pickle(output, sound + "_diarization.pkl")
    result = output.speaker_diarization
    # print the predicted speaker diarization
    for turn, speaker in result:
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Diarization model",
        default="models/speaker-diarization-community-1")
    parser.add_argument(
        "--sound", type=str, help="Sound file to process", default="audio.wav")
    args = parser.parse_args()

    turns = speakers(sound=args.sound, diarization_model=args.model, min_speakers=4, max_speakers=8)
