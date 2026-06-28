# ASR
Automated speech recognition (mostly for Russian).

See the pipeline in `process.sh`.

- `diarization.py` maps timestamps to speakers
- `speech2text.py` converts audio to text (with its own timestamps)
- `save_load_diarization.py` contains helper functions
