"""
ASR Module - Automatic Speech Recognition

This module provides a flexible ASR class for transcribing audio using Whisper models 
with configurable parameters.
"""
from transformers import pipeline
import numpy as np
import torchaudio
import base64
import torch


class ASR:
    """
    Automatic Speech Recognition class using Hugging Face Whisper models.
    
    This class handles audio preprocessing and transcription with configurable
    parameters for model size, device, and sample rate.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ASR module with given configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with the following optional keys:
                - asr_model (str): Whisper model variant ("tiny", "small", "medium", "large")
                - asr_device (str): Device to run inference on (e.g., "cuda:0", "cpu")
                - sample_rate (float): Audio sample rate to use (default: 16000.0)
                - max_audio_length (int): Maximum audio length in seconds (default: 60)
        """
        # Default configuration
        self.config = {
            "asr_model": "medium",
            "asr_device": "cuda:0",
            "sample_rate": 16000.0,
            "max_audio_length": 60
        }
        
        # Update with provided config
        if config is not None:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        # Set parameters from config
        self.AUDIO_SAMPLE_RATE = self.config["sample_rate"]
        self.MAX_INPUT_AUDIO_LENGTH = self.config["max_audio_length"]
        
        # Determine full model name based on variant
        model_variant = self.config["asr_model"]
        model_name = f"openai/whisper-{model_variant}"
        
        # Check device availability
        device = self.config["asr_device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Initialize the transcriber pipeline
        print(f"Initializing ASR with model: {model_name} on device: {device}")
        self.transcriber = pipeline(
            "automatic-speech-recognition", 
            model=model_name, 
            device=device
        )
        print("ASR initialization complete")
    
    def preprocess_audio(self, input_audio):
        """
        Preprocess audio file to the correct sample rate.
        
        Args:
            input_audio (str): Path to audio file
            
        Returns:
            tuple: Sample rate and numpy array of audio data
        """
        arr, org_sr = torchaudio.load(input_audio)
        new_arr = torchaudio.functional.resample(
            arr, 
            orig_freq=org_sr, 
            new_freq=self.AUDIO_SAMPLE_RATE
        )
        return self.AUDIO_SAMPLE_RATE, new_arr.numpy()
    
    def transcribe(self, audio_path, language=None, task="transcribe"):
        """
        Transcribe audio from a file path.
        
        Args:
            audio_path (str): Path to audio file
            language (str, optional): Language code (e.g., "en", "fr")
            task (str, optional): "transcribe" or "translate" (default: "transcribe")
            
        Returns:
            str: Transcribed text
        """
        sr, y = self.preprocess_audio(audio_path)
        y = y.astype(np.float32)[0, :]  # single channel only
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        
        generate_kwargs = {"task": task}
        if language is not None:
            generate_kwargs["language"] = language.lower()
        
        return self.transcriber(
            {"sampling_rate": sr, "raw": y}, 
            generate_kwargs=generate_kwargs
        )["text"]
    
    def transcribe_raw(self, audio_array, language=None, task="transcribe"):
        """
        Transcribe audio from a numpy array.
        
        Args:
            audio_array (numpy.ndarray): Audio array
            language (str, optional): Language code (e.g., "en", "fr")
            task (str, optional): "transcribe" or "translate" (default: "transcribe")
            
        Returns:
            str: Transcribed text
        """
        # Normalize audio
        y = audio_array.astype(np.float32)
        y /= np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        
        generate_kwargs = {"task": task}
        if language is not None:
            generate_kwargs["language"] = language.lower()
        
        return self.transcriber(
            {"sampling_rate": self.AUDIO_SAMPLE_RATE, "raw": y}, 
            generate_kwargs=generate_kwargs
        )["text"]
    
    def run_asr(self, request):
        """
        Process a request containing base64-encoded audio.
        
        Args:
            request: Object with source_language, target_language, and audio_base64 attributes
            
        Returns:
            str: Transcribed text
        """
        # Extract fields from the request
        source_language = request.source_language
        target_language = request.target_language
        audio_base64 = request.audio_base64
        
        print(f"Processing ASR request - Source: {source_language} | Target: {target_language}")
        
        # Decode the base64-encoded audio
        audio_data = base64.b64decode(audio_base64)
        
        # Convert the byte data to a NumPy array (int16)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Determine task based on target language
        task = "translate" if source_language != target_language else "transcribe"
        
        # Transcribe the audio
        return self.transcribe_raw(
            audio_array, 
            language=source_language.lower(),
            task=task
        )