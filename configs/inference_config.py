import torch

# Comprehensive configuration with both model paths and runtime settings
config = {
    "chat_type": "voice",  # ['text', 'visual_speech', "multimodal"]
    
    # Model paths
    "wav_config_path": "./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "wav_model_path": "./CHECKPOINTS/wavtokenizer_large_speech_320_24k.ckpt",
    "encoder_model_path": "charsiu/g2p_multilingual_byT5_tiny_16_layers_100",
    "tokenizer_path": "google/byt5-small",
    "gpt_checkpoint_path": "./CHECKPOINTS/ckpt_english_tiny.pt",
    
    # LLM settings
    #"llm_checkpoint": "meta-llama/Llama-3.1-8B-Instruct",
    #"llm_checkpoint": "microsoft/Phi-4-multimodal-instruct",
    #"llm_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct"
    "llm_device": "cuda:0",
    "llm_max_tokens": 1000,
    "llm_temperature": 0.7,
    "llm_top_p": 0.95,
    "llm_top_k": 40,
    
    # TTS model settings
    "tts_device_1": 1,  # GPU ID
    "tts_device_2": 2,
    
    # Streaming settings
    "system_prompt": "You are a friendly voicebot that answers questions in a concise way and do not use abbreviation.Give short responses",
    "initial_dump_size_1": 10,
    "initial_dump_size_2": 160,
    "max_dump_size": 1280,
    "max_audio_length": 8000,
    
    # Special tokens
    #"eos_token": "<|end|>",
    # "eos_token": "<|im_end|>",
    #"eos_token":"<eos>",
    "eos_token": "<|eot_id|>",
    "pad_token_id": 384,
    "eoa_token_id": 453,
    
    # API settings
    "api_host": "0.0.0.0",
    "api_port": 5003,
    
    # ASR settings
    "asr_model": "small",                # Whisper model variant: tiny, base, small, medium, large
    "asr_device": "cuda:2",               # Device for ASR model
    "asr_sample_rate": 16000.0,           # Audio sample rate in Hz
    "asr_max_audio_length": 60,           # Maximum audio length in seconds
    "asr_default_language": "english",         # Default language for transcription
    "asr_enable_translation": False,      # Enable translation instead of transcription by default
}