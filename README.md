## LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

**[Sambal Shikar](https://github.com/mbzuai-oryx/LLMVoX?tab=readme-ov-file)**, **[Mohammed Irfan K](https://scholar.google.com/citations?user=GJp0keYAAAAJ&hl=en)**, **[Sahal Shaji Mullappilly](https://scholar.google.com/citations?user=LJWxVpUAAAAJ&hl=en)**, **[Fahad Khan](https://sites.google.com/view/fahadkhans/home)**, **[Jean Lahoud](https://scholar.google.com/citations?user=LsivLPoAAAAJ&hl=en)**, **[Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ)**, **[Salman Khan](https://salman-h-khan.github.io/)**, **[Hisham Cholakkal](https://scholar.google.com/citations?hl=en&user=bZ3YBRcAAAAJ)**

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
- 🚀 **Lightweight & Fast**: Only 30M parameters, delivering speech with end-to-end latency as low as 300ms
- 🔌 **LLM-Agnostic**: Just plug with any existing LLM and Vision-Language Models without requiring fine-tuning or architectural modifications.
- 🌊 **Multi-Queue Streaming**: Enables continuous, low-latency speech generation and infinite-length dialogues
- 🌐 **Multilingual Support**: Easily adaptable to new languages with only dataset adaptation

Installation
Requirements
bashCopy# Clone the repository
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
Quick Start
Configuration Basics
LLMVoX requires a few base paths to be set correctly:

wav_config_path: Path to WavTokenizer configuration file
wav_model_path: Path to the pretrained WavTokenizer model checkpoint
encoder_model_path: Path to the G2P model for phonetic embeddings
tokenizer_path: Path to the ByT5 tokenizer
gpt_checkpoint_path: Path to the trained LLMVoX model checkpoint

Running with Different Configurations
Voice Chat Configuration Guide
LLMVoX supports voice-based conversations through its streaming server. Here's how to configure and use the voice chat functionality:
Basic Usage
bashCopypython streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct"
Configuration Parameters Explained
GPU Resource Allocation
LLMVoX uses a multi-queue approach with two TTS model replicas. You can specify which GPUs to use:
bashCopy# Run TTS models on separate GPUs
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --tts_device_1 1 --tts_device_2 2

# Or run both on the same GPU (if memory allows)
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --tts_device_1 0 --tts_device_2 0

# Specify GPU for LLM separately
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --llm_device "cuda:0" --tts_device_1 1 --tts_device_2 2
Streaming Chunk Size Parameters
Control the balance between latency and quality:
bashCopy# Lower latency setup (faster initial response but potentially lower quality)
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --initial_dump_size_1 5 --initial_dump_size_2 40 --max_dump_size 320

# Higher quality setup (slightly higher latency but better speech)
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --initial_dump_size_1 20 --initial_dump_size_2 320 --max_dump_size 2560

# Default balanced setup
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --initial_dump_size_1 10 --initial_dump_size_2 160 --max_dump_size 1280

initial_dump_size_1: Number of speech tokens for the first chunk (smaller = faster first response)
initial_dump_size_2: Initial chunk size for the second TTS model (can be larger as it runs while first chunk plays)
max_dump_size: Maximum chunk size that the system will scale up to (larger = better quality)

LLM-Specific Parameters
Different LLMs use different end-of-sequence tokens:
bashCopy# For LLaMA models
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --eos_token "<|eot_id|>"

# For Mistral models
python streaming_server.py --chat_type voice --llm_checkpoint "mistralai/Mistral-7B-Instruct-v0.2" --eos_token "<|im_end|>"

# For other models (check your model's documentation)
python streaming_server.py --chat_type voice --llm_checkpoint "your-model-name" --eos_token "<|end|>"
ASR Configuration (for Speech Input)
LLMVoX uses Whisper for converting speech to text:
bashCopy# Use a larger Whisper model for better transcription
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --asr_model "medium" --asr_device "cuda:3"

# Use a smaller model for faster processing
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --asr_model "tiny" --asr_device "cuda:0"
System Prompt Customization
Control the LLM's response style:
bashCopy# For concise responses
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --system_prompt "You are a friendly voicebot that answers questions in a concise way and do not use abbreviation. Keep responses brief."

# For more detailed explanations
python streaming_server.py --chat_type voice --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --system_prompt "You are a helpful AI assistant that provides detailed, thorough explanations. Avoid abbreviations when speaking."
Complete Example
Here's a complete example with all key parameters configured:
bashCopypython streaming_server.py \
  --chat_type voice \
  --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" \
  --llm_device "cuda:0" \
  --tts_device_1 1 \
  --tts_device_2 2 \
  --asr_model "small" \
  --asr_device "cuda:3" \
  --initial_dump_size_1 10 \
  --initial_dump_size_2 160 \
  --max_dump_size 1280 \
  --max_audio_length 8000 \
  --eos_token "<|eot_id|>" \
  --system_prompt "You are a friendly voicebot that answers questions concisely without abbreviations."
How it Works
When you run voice chat:

The ASR model transcribes your speech input
The LLM generates a response text stream
Two LLMVoX instances alternate processing text chunks at sentence boundaries
Initial chunks are smaller for faster response, while later chunks are larger for better quality
Audio is played in real-time while the rest of the response is still being generated

This multi-queue architecture enables both low latency (as fast as 475ms) and high-quality speech output.
Text Chat (Text-to-Speech)
bashCopy# Basic text chat with LLaMA 3.1 8B
python streaming_server.py --chat_type text --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --llm_device "cuda:0" --tts_device_1 1 --tts_device_2 2

# Customize LLM generation parameters
python streaming_server.py --chat_type text --llm_checkpoint "meta-llama/Llama-3.1-8B-Instruct" --llm_temperature 0.5 --llm_top_p 0.9 --llm_top_k 30
Visual Speech (Speech + Image → Speech)
bashCopy# Using Qwen 2.5 VL as the vision-language model
python streaming_server.py --chat_type visual_speech --llm_checkpoint "Qwen/Qwen2.5-VL-7B-Instruct" --llm_device "cuda:0" --tts_device_1 1 --tts_device_2 2

# Using Phi-4 Multimodal
python streaming_server.py --chat_type visual_speech --llm_checkpoint "microsoft/Phi-4-multimodal-instruct" --llm_device "cuda:0"
Multimodal Chat (Audio + Image → Speech)
bashCopy# Using Qwen 2.5 VL
python streaming_server.py --chat_type multimodal --llm_checkpoint "Qwen/Qwen2.5-VL-7B-Instruct" --llm_device "cuda:0" --tts_device_1 1 --tts_device_2 2

# Using LLaVA
python streaming_server.py --chat_type multimodal --llm_checkpoint "llava-hf/llava-1.5-7b-hf" --llm_device "cuda:0"
APIs
The streaming server exposes several API endpoints:

/tts: For text-to-speech conversion
/voicechat: For voice-based conversations
/multimodalchat: For multimodal interactions
/vlmschat: For visual speech interactions

Training Your Own Model
To train LLMVoX on your own data:
bashCopy# Single GPU training
python train.py --batch_size=2 --compile=True

# Distributed training on multiple GPUs
torchrun --standalone --nproc_per_node=4 train.py
Citation
If you find our work useful, please consider citing:
bibtexCopy@misc{shikhar2025llmvoxautoregressivestreamingtexttospeech,
    title={LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM},
    author={Sambhal Shikhar and Mohammed Irfan Kurpath and Sahal Shaji Mullappilly and Jean Lahoud and Fahad Khan and Rao Muhammad Anwer and Salman Khan and Hisham Cholakkal},
    year={2025},
    eprint={2503.04724},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.04724}
}
Acknowledgments
We thank the reviewers and colleagues who provided valuable feedback on this work. We also acknowledge the open-source contributions that made this project possible:

Andrej Karpathy's NanoGPT - Training code for LLMVoX is based on this repository
WavTokenizer - For audio tokenization
Whisper - Used for ASR in our pipeline
FastAPI - For creating our streaming server API
ByT5 - For the multilingual phoneme embeddings
Hugging Face Transformers - For easy integration with various LLMs

License
This project is licensed under the MIT License - see the LICENSE file for details.