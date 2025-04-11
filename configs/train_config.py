"""
Configuration setting for training
"""
import torch
import os
from dotenv import load_dotenv
load_dotenv()

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
    # have to change to bnt5 "Virus-Proton/ByT5_Bengali_IPA", #
    "encoder_model_path": 'charsiu/g2p_multilingual_byT5_tiny_16_layers_100',
    "tokenizer_path": "Virus-Proton/ByT5_Bengali_IPA", #'google/byt5-small',  # same as encoder
    "checkpoint_filename": 'ckpt.pt',
    "wav_config_path": "WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    # "/nvme-data/sambal/nanoGPT/wavtokenizer_large_speech_320_24k.ckpt",
    "wav_model_path": "./CHECKPOINTS/wavtokenizer_large_speech_320_24k.ckpt",

    # ===== Training Control =====
    "is_train": True,
    "out_dir": "results",
    "eval_interval": 100,  # 1000
    "log_interval": 1,
    "eval_iters": 1,
    "eval_only": False,
    "always_save_checkpoint": False,
    "init_from": 'gpt2', #['scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    "compile": True,

    # ===== Logging =====
    "wandb_log": False,
    "wandb_project": 'LLMVoX_Bn',
    "wandb_run_name": "first_test_run",
    "wandb_token": os.getenv("WANDB_APWANDB_TOKENI_KEY"),

    # ===== Dataset =====
    "data_path": "./data/virutalassistant/train_data.json", # "/nvme-data/sambal/valid_data_400k.json",
    "speech_data_folder": "./data/virutalassistant/audios",# "/nvme-data/sambal/audio_files",

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
    "warmup_iters": 500,  # 50000
    "lr_decay_iters": 1000,  # 2600000
    "min_lr": 3e-6,
}
