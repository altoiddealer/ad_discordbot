# config.py
# Replace with your actual Discord bot token and text channel ID where transcriptions should be sent.
TOKEN = ""

WRAP_MESSAGES = False # Set to True to wrap messages, False for individual messages

ASR_ENGINE = "whisper"  # "whisper", "omnisense" or "faster_whisper"

# Whisper configuration
WHISPER_MODEL_NAME = "base" # base, large, small,...
WHISPER_CUDA_VISIBLE_DEVICES = "0"  # GPU id to use

# OmniSenseVoice configuration
OMNISENSE_LANGUAGE = "auto"  # auto, zh, en, yue, ja, ko 
OMNISENSE_TEXTNORM = "withitn"  # withitn or woitn
OMNISENSE_DEVICE_ID = 0  # -1 for CPU, 0+ for GPU
OMNISENSE_QUANTIZE = True  # Use quantized model
OMNISENSE_MODEL_DIR = "SenseVoiceSmall"  # None for default

# Faster-Whisper configuration
FASTER_WHISPER_MODEL_SIZE = "base"  # Should be same as for Whisper, check faster_whisper documentation
FASTER_WHISPER_DEVICE = "cuda"  # I'll let you use cpu this time,"cuda" or "cpu"
FASTER_WHISPER_COMPUTE_TYPE = "int8_float16"  # Compute type: "float16" (GPU), "int8_float16" (GPU), "int8" (CPU)

class AudioConfig:
    # Audio processing settings
    SPEECH_VOLUME_THRESHOLD = 1 # Not recommended to change
    CHUNK_SILENCE_THRESHOLD = 0.5  # Seconds to process a chunk
    FINAL_SILENCE_THRESHOLD = 0.8  # Seconds to finalize transcription
    SAMPLE_RATE = 48000
    CHANNELS = 2
    SAMPLE_WIDTH = 2  # 16-bit PCM
    
    # Volume normalization parameters, adjust if needed
    TARGET_RMS = 1400
    MAX_GAIN = 10
    MIN_RMS = 100