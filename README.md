# Deep-learning-VAD
An AI-based Voice Activity Detection (VAD) system designed to accurately identify human speech in noisy environments using machine learning and digital signal processing techniques. The project aims to distinguish speech signals from background noise, improving speech detection accuracy and robustness in real-world audio applications.

import sounddevice as sd
import numpy as np
import webrtcvad
import collections

# Initialize Voice Activity Detector
vad = webrtcvad.Vad()

# Aggressiveness level (0 to 3)
vad.set_mode(2)

# Audio parameters
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # milliseconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)

print("AI-Based Voice Activity Detection")
print("Speak into the microphone...")
print("Press Ctrl+C to stop\n")

def detect_voice(indata, frames, time, status):
    if status:
        print(status)

    # Convert audio to 16-bit PCM
    audio = (indata[:, 0] * 32767).astype(np.int16)

    # Process audio frames
    for i in range(0, len(audio) - FRAME_SIZE + 1, FRAME_SIZE):
        frame = audio[i:i + FRAME_SIZE]

        audio_bytes = frame.tobytes()

        # Detect speech
        is_speech = vad.is_speech(audio_bytes, SAMPLE_RATE)

        if is_speech:
            print("Voice Detected")
        else:
            print("Background Noise / Silence")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=detect_voice,
        blocksize=FRAME_SIZE
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nVoice Activity Detection Stopped")
    
