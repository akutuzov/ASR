#! /bin/bash

RECORDING=${1}  # e.g., recording.wav
SOUND=${2}  # e.g., normalized_recording.wav

# To record group calls in messengers, first check your audio devices with:
# pactl list short sources

# Mark what your main microphone source (input) and main playing source (output) are
# Record with something like
ffmpeg -f pulse -i alsa_output.usb-DisplayLink_ThinkPad_Hybrid_USB-C_with_USB-A_Dock_11891881-02.analog-stereo.monitor -f pulse -i alsa_input.usb-_Webcam_C110-02.mono-fallback -filter_complex amix=inputs=2 ${RECORDING}

# The resulting file can be wav or mp3, ffmpeg handles encoding transparently.

# Then we normalize the recoding with sox:
sox ${RECORDING} -r 16k ${SOUND} norm -0.5 compand 0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2

# Finally, ASR:
python3 speech2text.py --sound ${SOUND}

# If you already have the transcription (in json), you can run only diarization and post-processing:
# python3 speech2text.py --sound ${SOUND} --transcription asr_ready.json

# If you also have the diarization ready (in pkl), you can run only post-processing:
# python3 speech2text.py --sound ${SOUND} --transcription asr_ready.json --diarization diar.pkl