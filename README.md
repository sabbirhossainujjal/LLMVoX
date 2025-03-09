## LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM

<div style="width:100%; height:8px; background: linear-gradient(to right, #ffd700, #ff6347, #9370db, #3cb371, #1e90ff, #ff69b4);"></div>

**[Sambal Shikar](https://github.com/sambaI)\***, **[Mohammed Irfan K](https://scholar.google.com/citations?user=GJp0keYAAAAJ&hl=en)**, **[Sahal Shaji Mullappilly](https://scholar.google.com/citations?user=LJWxVpUAAAAJ&hl=en)\***, **[Fahad Khan](https://sites.google.com/view/fahadkhans/home)**, **[Jean Lahoud](https://scholar.google.com/citations?user=LsivLPoAAAAJ&hl=en)**, **[Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ)**, **[Salman Khan](https://salman-h-khan.github.io/)**, **[Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ)**

Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI), UAE

<div>
<a href="https://mbzuai-oryx.github.io/LLMVoX/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2503.04724"><img src="https://img.shields.io/badge/arXiv-2503.04724-b31b1b.svg" alt="arXiv"></a>
<a href="https://github.com/mbzuai-oryx/LLMVoX/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<p align="center">
    <img src="assets/arch_diagram.svg" alt="LLMVoX Architecture" width="800px">
</p>

## Overview

LLMVoX is a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming Text-to-Speech (TTS) system designed to convert text outputs from Large Language Models into high-fidelity streaming speech with low latency. Our approach achieves significantly lower Word Error Rate compared to speech-enabled LLMs while operating at comparable latency and speech quality.

Key features:
- 🚀 **Lightweight & Fast**: Only 30M parameters, delivering speech with minimal latency
- 🔌 **LLM-Agnostic**: Works with any LLM without requiring fine-tuning or architectural modifications
- 🌊 **Multi-Queue Streaming**: Enables continuous, low-latency speech generation and infinite-length dialogues
- 🌐 **Multilingual Support**: Easily adaptable to new languages with only dataset adaptation
- 🧩 **Plug-and-Play Design**: Seamlessly integrates with various LLMs and Vision-Language Models

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/mbzuai-oryx/LLMVoX.git
cd LLMVoX

# Create and activate a conda environment
conda create -n llmvox python=3.9
conda activate llmvox

# Install dependencies
pip install -r requirements.txt

# Download checkpoints (if not already in the repository)
mkdir -p CHECKPOINTS
# Download wavtokenizer_large_speech_320_24k.ckpt and ckpt_english_tiny.pt
# and place them in the CHECKPOINTS directory
```

## Quick Start

### Running the Streaming Server

LLMVoX provides a streaming server that can be run with different configurations:

```bash
# Voice chat - for speech-to-speech conversation
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct"

# Text chat - for text-to-speech conversion
python streaming_server.py --chat_type text --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct"

# Visual speech - for speech interaction with visual inputs
python streaming_server.py --chat_type visual_speech --llm_checkpoint "Qwen/Qwen2.5-VL-7B-Instruct"

# Multimodal chat - for handling multiple modalities (audio, images, text)
python streaming_server.py --chat_type multimodal --llm_checkpoint "microsoft/Phi-4-multimodal-instruct"
```

You can also customize other parameters:

```bash
# Customize streaming parameters
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" \
    --initial_dump_size_1 20 --initial_dump_size_2 40 --max_dump_size 160

# Change API server settings
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" \
    --api_host "127.0.0.1" --api_port 8000
```

### APIs

The streaming server exposes several API endpoints:

- `/tts`: For text-to-speech conversion
- `/voicechat`: For voice-based conversations
- `/multimodalchat`: For multimodal interactions
- `/vlmschat`: For visual speech interactions

### Training Your Own Model

To train LLMVoX on your own data:

```bash
# Single GPU training
python train.py --batch_size=2 --compile=True

# Distributed training on multiple GPUs
torchrun --standalone --nproc_per_node=4 train.py
```



## Architecture

<p align="center">
    <img src="assets/arch_diagram.svg" alt="LLMVoX Architecture" width="800px">
</p>

LLMVoX uses a decoder-only Transformer architecture that autoregressively predicts discrete speech tokens from streaming LLM text. The key components include:

1. **Neural Audio Tokenization**: Uses WavTokenizer to discretize continuous audio waveforms
2. **Byte-Level Grapheme-to-Phoneme Embedding**: Employs ByT5-based G2P model for phonetic information
3. **Decoder-only Transformer**: A lightweight 4-layer transformer that generates speech tokens
4. **Multi-Queue Streaming**: Enables continuous and potentially infinite-length speech generation



## Configuration

LLMVoX provides extensive configuration options for both training and inference:

### Inference Configuration

Key parameters in the inference configuration:
- `chat_type`: Supported modes include 'text', 'visual_speech', and 'multimodal'
- `llm_checkpoint`: Specify the LLM to use (e.g., "meta-llama/Llama-3.1-8B-Instruct")
- `tts_device_1` and `tts_device_2`: GPU IDs for the TTS models
- `initial_dump_size_1` and `initial_dump_size_2`: Initial chunk sizes for speech generation
- `max_dump_size`: Maximum chunk size for better speech quality
- `system_prompt`: Default prompt for the LLM

### Training Configuration

Key parameters in the training configuration:
- `n_layer`, `n_head`, `n_embd`: Model architecture parameters
- `block_size`: Context size for attention
- `batch_size` and `gradient_accumulation_steps`: Training batch parameters
- `learning_rate` and related parameters: Optimization settings
- `data_path` and `speech_data_folder`: Paths to training data

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{shikhar2025llmvoxautoregressivestreamingtexttospeech,
    title={LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM},
    author={Sambhal Shikhar and Mohammed Irfan Kurpath and Sahal Shaji Mullappilly and Jean Lahoud and Fahad Khan and Rao Muhammad Anwer and Salman Khan and Hisham Cholakkal},
    year={2025},
    eprint={2503.04724},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.04724}
}
```





## Acknowledgments

We thank the reviewers and colleagues who provided valuable feedback on this work. We also acknowledge the open-source contributions that made this project possible:

- [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) - Training code for LLMVoX is based on this repository
- [WavTokenizer](https://github.com/shengbaj/wavtokenizer) - For audio tokenization
- [Whisper](https://github.com/openai/whisper) - Used for ASR in our pipeline
- [FastAPI](https://github.com/tiangolo/fastapi) - For creating our streaming server API
- [ByT5](https://github.com/google-research/text-to-text-transfer-transformer) - For the multilingual phoneme embeddings
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - For easy integration with various LLMs