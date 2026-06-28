import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="ASR model",
        default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--sound", type=str, help="Sound file to process", default="audio.wav")
    parser.add_argument(
        "--transcription", type=str, help="Ready transcription (JSON)")
    parser.add_argument(
        "--diarization", type=str, help="Ready diarization (pkl)")
    parser.add_argument(
        "--transcribe_only", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--min_speakers", type=int, help="Minimal number of speakers (for diarization)", default=2)
    parser.add_argument(
        "--max_speakers", type=int, help="Maximum number of speakers (for diarization)", default=6)
    parser.set_defaults(transcribe_only=False)
    args = parser.parse_args()

    modelname = args.model

    soundfile = args.sound

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        setattr(torch.distributed, "is_initialized", lambda: False)  # monkey patching
    device = torch.device(device)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if not args.transcribe_only:
        from diarization import speakers
        from save_load_diarization import load_diarization_pickle
        if args.diarization:
            turns = load_diarization_pickle(args.diarization).speaker_diarization
            print(f"Loaded diarization from {args.diarization}")
        else:
            print("Diarization in progress...")
            turns = speakers(sound=soundfile, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        print("Diarization complete")

        unique_speakers = set()

        for turn, speaker in turns:
            unique_speakers.add(speaker)
        print(f"Speakers: {len(unique_speakers)}")

        print(f"Turns: {len(turns)}")

    if args.transcription:
        with open(args.transcription, "r") as f:
            transcription = json.load(f)
        print(f"Loaded transcription from {args.transcription}")
    else:
        print("Transcribing...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            modelname,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa"
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(modelname)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_dtype,
            return_timestamps=True
        )

        transcription = pipe(
            soundfile, generate_kwargs={"language": "russian", "return_timestamps": True}
        )

    print(f"Chunks: {len(transcription['chunks'])}")

    if not args.transcribe_only:
        utterances = []
        last_speaker = None

        for chunk in transcription["chunks"]:
            chunk_speakers = set()
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            if not end:
                end = start + 10
                print(f"Replaced null timestamp with {end}")
            for turn in turns:
                turn_times, speaker = turn
                if (turn_times.start >= start and turn_times.end <= end) or (
                        start >= turn_times.start and end <= turn_times.end):
                    chunk_speakers.add(speaker)
            if not chunk_speakers:
                chunk_speakers = last_speaker

            speaker_str = ", ".join(sorted(chunk_speakers)) if chunk_speakers else "UNKNOWN"
            utterances.append(f"{speaker_str}: {chunk['text']}")
            last_speaker = chunk_speakers

        # Final postprocessing:
        previous_speaker = None
        for line in utterances:
            el = line.strip()
            if el:
                if el.endswith(":"):
                    continue
                speaker, phrase = el.split(": ")
                if previous_speaker:
                    if speaker == previous_speaker:
                        print(phrase, end="")
                    else:
                        print("\n")
                        print(el, end="")
                else:
                    print(el, end="")
                previous_speaker = speaker
        print("\n")

    if not args.transcription:
        with open(soundfile + "_transcribed.json", "w") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=4)
        print(f"Saved transcription to {soundfile + "_ready.json"} (JSON format)")
