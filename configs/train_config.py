"""
Configuration settings for speech prediction model.
Includes parameters for both training and inference.
"""
import torch

# Comprehensive configuration with both training and inference settings
config = {
    # ===== System Settings =====
    "device": 'cuda:0',
    "dtype": 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    "backend": 'nccl',
    
    # ===== Model Architecture =====
    "n_layer": 4,
    "n_head": 8,
    "n_embd": 768,
    "block_size": 4096*2,
    "dropout": 0.0,
    "bias": False,
    
    # ===== Model Paths =====
    "encoder_model_path": 'charsiu/g2p_multilingual_byT5_tiny_16_layers_100',
    "tokenizer_path": 'google/byt5-small',
    "checkpoint_filename": 'ckpt.pt',
    "wav_config_path": "WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "wav_model_path": "/nvme-data/sambal/nanoGPT/wavtokenizer_large_speech_320_24k.ckpt",
    
    # ===== Training Control =====
    "is_train": True,
    "out_dir": "exp_1",
    "eval_interval": 1000,
    "log_interval": 1,
    "eval_iters": 1,
    "eval_only": False,
    "always_save_checkpoint": True,
    "init_from": 'resume',
    "compile": True,
    
    # ===== Logging =====
    "wandb_log": True,
    "wandb_project": 'speech_stream',
    "wandb_run_name": "out_4_layers_arabic_number_finetune",
    
    # ===== Dataset =====
    "data_path": "/nvme-data/sambal/valid_data_400k.json",
    "speech_data_folder":"/nvme-data/sambal/audio_files",
    
    # ===== Training Hyperparameters =====
    "gradient_accumulation_steps": 4,
    "batch_size": 2,
    "learning_rate": 3e-4,
    "max_iters": 2600000,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    
    # ===== Learning Rate Schedule =====
    "decay_lr": True,
    "warmup_iters": 50000,
    "lr_decay_iters": 2600000,
    "min_lr": 3e-6,
}