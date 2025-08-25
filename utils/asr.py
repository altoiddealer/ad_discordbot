import torch
import os
import config_stt as config
import threading

class ASRManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.whisper_model = None
            self.omnisense_model = None
            self.faster_whisper_model = None  # Added Faster-Whisper support
            self.lock = threading.Lock()
            self._initialized = True

    def initialize(self):
        with self.lock:
            if config.ASR_ENGINE == "whisper" and not self.whisper_model:
                import whisper
                os.environ["CUDA_VISIBLE_DEVICES"] = config.WHISPER_CUDA_VISIBLE_DEVICES
                self.whisper_model = whisper.load_model(
                    config.WHISPER_MODEL_NAME, 
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif config.ASR_ENGINE == "omnisense" and not self.omnisense_model:
                from omnisense.models.sensevoice import OmniSenseVoiceSmall
                self.omnisense_model = OmniSenseVoiceSmall(
                    model_dir=config.OMNISENSE_MODEL_DIR,
                    device_id=config.OMNISENSE_DEVICE_ID,
                    quantize=config.OMNISENSE_QUANTIZE
                )
            elif config.ASR_ENGINE == "faster_whisper" and not self.faster_whisper_model:
                from faster_whisper import WhisperModel
                self.faster_whisper_model = WhisperModel(
                    config.FASTER_WHISPER_MODEL_SIZE,
                    device=config.FASTER_WHISPER_DEVICE,
                    compute_type=config.FASTER_WHISPER_COMPUTE_TYPE
                )

    def transcribe(self, audio_file_path):
        self.initialize()  # Ensure models are loaded
        with self.lock:
            if config.ASR_ENGINE == "whisper":
                return self._whisper_transcribe(audio_file_path)
            elif config.ASR_ENGINE == "omnisense":
                return self._omnisense_transcribe(audio_file_path)
            elif config.ASR_ENGINE == "faster_whisper":
                return self._faster_whisper_transcribe(audio_file_path)
            else:
                raise ValueError(f"Unsupported ASR engine: {config.ASR_ENGINE}")

    def _whisper_transcribe(self, audio_file_path):
        result = self.whisper_model.transcribe(audio_file_path)
        return result["text"]

    def _omnisense_transcribe(self, audio_file_path):
        try:
            results = self.omnisense_model.transcribe(
                audio_file_path,
                language=config.OMNISENSE_LANGUAGE,
                textnorm=config.OMNISENSE_TEXTNORM
            )
            return " ".join([result.text for result in results if result.text])
        except Exception as e:
            print(f"OmniSenseVoice Error: {str(e)}")
            return ""

    def _faster_whisper_transcribe(self, audio_file_path):
        try:
            segments, _ = self.faster_whisper_model.transcribe(audio_file_path)
            transcription = " ".join([segment.text for segment in segments])
            return transcription
        except Exception as e:
            print(f"Faster-Whisper Error: {str(e)}")
            return ""

# Singleton instance
asr_manager = ASRManager()
