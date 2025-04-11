"""
Data loading and processing utilities for speech prediction model training.
"""
from WavTokenizer.decoder.pretrained import WavTokenizer
from WavTokenizer.encoder.utils import convert_audio
import os
import sys
import json
import torch
import numpy as np
import librosa
import itertools
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Sequence, List
from torch.utils.data import Dataset, DataLoader
import transformers

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now you can import from WavTokenizer
# Constants
IGNORE_INDEX = -100
PAD_TOKEN_ID = 384
EOA_TOKEN_ID = 453  # End of audio token ID


class SpeechDataset(Dataset):
    """Dataset for loading and preprocessing speech and text data."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, device='cpu',
                 data_path=None, speech_folder_path=None, wav_config_path=None, wav_model_path=None):
        """
        Initialize the speech dataset.

        Args:
            tokenizer: Tokenizer for processing text
            device: Device to run processing on ('cpu' or 'cuda')
            data_path: Path to the JSON data file
            wav_config_path: Path to the WavTokenizer config file
            wav_model_path: Path to the WavTokenizer model checkpoint
        """
        super(SpeechDataset, self).__init__()

        # Default paths if not provided
        if data_path is None:
            data_path = "/nvme-data/sambal/LLMVoX/data.json"
        if speech_folder_path is None:
            speech_folder_path = "/nvme-data/sambal/LLMVoX/audio_files"
        if wav_config_path is None:
            wav_config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        if wav_model_path is None:
            wav_model_path = "./CHECKPOINTS/wavtokenizer_large_speech_320_24k.ckpt"

        # Load dataset
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.device = device
        self.speech_folder_path = speech_folder_path
        # Initialize WavTokenizer
        print(f"Loading WavTokenizer from {wav_model_path}")
        self.wavtokenizer = WavTokenizer.from_pretrained0802(
            wav_config_path, wav_model_path)
        self.wavtokenizer = self.wavtokenizer.to(self.device)
        print("WavTokenizer loaded successfully")

    def __len__(self):
        return len(self.list_data_dict)

    def process_speech(self, speech_path):
        """
        Process speech audio file to extract features and tokens.

        Args:
            speech_path: Path to the speech audio file

        Returns:
            features: Processed speech features
            discrete_code: Discrete speech tokens
            eoa_features: End of audio features
        """
        # Load audio file
        wav, sr = librosa.load(speech_path, sr=None, mono=True)
        wav = np.array(wav)
        wav = np.expand_dims(wav, 0)
        wav = torch.tensor(wav)

        # Standardize audio format
        wav = convert_audio(wav, sr, 24000, 1)
        bandwidth_id = torch.tensor([0])
        wav = wav.to(self.device)

        # Extract features and tokens using WavTokenizer
        features, discrete_code = self.wavtokenizer.encode_infer(
            wav, bandwidth_id=bandwidth_id)

        # Process discrete codes
        discrete_code = discrete_code[discrete_code !=
                                      EOA_TOKEN_ID].unsqueeze(0).unsqueeze(0)
        features = self.wavtokenizer.codes_to_features(discrete_code)

        # Create end-of-audio token and features
        eoa_token = torch.tensor([[EOA_TOKEN_ID]]).unsqueeze(0).to(self.device)
        eoa_features = self.wavtokenizer.codes_to_features(eoa_token)

        # Prepare final discrete code sequence
        discrete_code = torch.cat((discrete_code, eoa_token), dim=2)
        discrete_code = discrete_code.squeeze(1)

        # Prepare feature sequence
        zero_tensor = torch.zeros(1, 512, 1).to(self.device)
        features = torch.cat((zero_tensor, features), dim=2)
        features = features.permute(0, 2, 1)

        return features, discrete_code, eoa_features

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a single data item by index.

        Args:
            idx: Index of the data item

        Returns:
            data_dict: Dictionary containing processed data
        """
        item = self.list_data_dict[idx]
        data_dict = {}

        text_resp = item["answer_text"]
        speech_file = f"{item['id']}.wav"
        # Process speech audio
        speech_path = os.path.join(self.speech_folder_path, speech_file)

        features, discrete_code, eoa_feat = self.process_speech(speech_path)

        # Process text
        text_resp = text_resp.split(" ")
        out = self.tokenizer(text_resp)['input_ids']
        out = list(itertools.chain(*out)) + [385]  # Add end token
        text_tokens = torch.tensor(out)

        # Prepare data dictionary
        data_dict['text_tokens'] = text_tokens
        data_dict['speech_out_tokens'] = discrete_code.squeeze(0)

        data_dict['speech_embeddings'] = features.squeeze(0)
        data_dict["eo_feat"] = eoa_feat.squeeze(0)

        return data_dict


@dataclass
class SpeechDataCollator:
    """Collator for batching speech data examples."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            instances: List of data dictionaries

        Returns:
            batch: Batched and padded data
        """
        # Extract and pad text tokens
        labels = [instance['text_tokens'] for instance in instances]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=PAD_TOKEN_ID
        )

        # Truncate to maximum model length
        labels = labels[:, :self.tokenizer.model_max_length]

        # Handle single example case
        if len(instances) == 1:
            labels = labels.squeeze(0)

        # Create batch dictionary
        batch = {'labels': labels}

        # Process speech embeddings
        if 'speech_embeddings' in instances[0]:
            self._process_speech_embeddings(instances, batch)

        # Process speech output tokens
        if 'speech_out_tokens' in instances[0]:
            self._process_speech_tokens(instances, batch)

        return batch

    def _process_speech_embeddings(self, instances, batch):
        """Process and pad speech embeddings."""
        speech_embeddings_list = [
            instance['speech_embeddings'] for instance in instances]
        eo_feat_list = [instance['eo_feat'] for instance in instances]

        # Find max sequence length
        max_seq_len = max(se.shape[0] for se in speech_embeddings_list)

        # Pad each speech embedding
        padded_speech_embeddings = []
        for se, eo_feat in zip(speech_embeddings_list, eo_feat_list):
            eo_feat = eo_feat.squeeze()
            seq_len = se.shape[0]
            pad_len = max_seq_len - seq_len

            if pad_len > 0:
                # Pad with EoA feature
                pad = eo_feat.unsqueeze(0).repeat(pad_len, 1)
                se_padded = torch.cat([se, pad], dim=0)
            else:
                se_padded = se

            padded_speech_embeddings.append(se_padded)

        batch['speech_embeddings'] = torch.stack(padded_speech_embeddings)

    def _process_speech_tokens(self, instances, batch):
        """Process and pad speech output tokens."""
        speech_tokens_list = [instance['speech_out_tokens']
                              for instance in instances]

        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            speech_tokens_list,
            batch_first=True,
            padding_value=-1000  # Padding value for speech tokens
        )

        batch['speech_out_tokens'] = padded_tokens


def construct_speech_decoder_input(data, llm_model):
    """
    Construct input for the speech decoder by combining text and speech embeddings.

    Args:
        data: Batch data dictionary
        llm_model: Language model for generating text embeddings

    Returns:
        combined_embeddings: Combined and normalized embeddings
    """
    speech_embeddings = data['speech_embeddings']
    text_label = data['labels']
    print(f"Debugging: Speech Embeddings Shape: {speech_embeddings.size()}")
    # Ensure text_label is 2D
    if len(text_label.size()) == 1:
        text_label = text_label.unsqueeze(0)

    batch_size = speech_embeddings.size(0)
    text_label_len = text_label.size(1)

    # Create padding token
    pad_token = torch.tensor([PAD_TOKEN_ID]).unsqueeze(0).to(text_label.device)

    # Calculate padding needed
    num_repeats = speech_embeddings.size(1) - text_label_len

    # Adjust text label length to match speech embeddings
    if num_repeats < 0:
        padded_labels = text_label[:, :speech_embeddings.size(1)]
    else:
        pad_tokens = pad_token.repeat(batch_size, num_repeats)
        padded_labels = torch.cat([text_label, pad_tokens], dim=-1)

    # Move to the correct device
    padded_labels = padded_labels.to(speech_embeddings.device)

    # Get text embeddings using the language model
    text_label_embeddings = llm_model(padded_labels)
    print(f"Debugging: Text Label Embeddings Shape: {text_label_embeddings.size()}")
    # Combine text and speech embeddings
    speech_embeddings = speech_embeddings.to(speech_embeddings.device)
    combined_embeddings = torch.cat(
        [text_label_embeddings, speech_embeddings], dim=2)

    # Normalize the combined embeddings
    combined_embeddings = F.normalize(
        combined_embeddings, p=2, dim=2, eps=1e-8)

    return combined_embeddings


def get_batch(train_dataloader, context_length, device, device_type, llm_model):
    """
    Get a batch of data from the dataloader.

    Args:
        train_dataloader: DataLoader instance
        device: Device to load data onto
        device_type: Type of device ('cpu' or 'cuda')
        llm_model: Language model for generating embeddings

    Returns:
        x: Input embeddings
        y: Target speech tokens
    """
    # Get next batch
    data = next(iter(train_dataloader))

    # Construct decoder input
    X = construct_speech_decoder_input(data, llm_model)
    Y = data["speech_out_tokens"]

    # Skip batches that are too large
    if X.size(1) > int(context_length):
        data = next(iter(train_dataloader))
        X = construct_speech_decoder_input(data, llm_model)
        Y = data["speech_out_tokens"]

    # Move data to device
    if device_type == 'cuda':
        # Non-blocking transfer for better performance
        x, y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    else:
        x, y = X.to(device), Y.to(device)

    return x, y


def create_data_module(tokenizer, device, config=None):
    """
    Create a data module with dataset and collator.

    Args:
        tokenizer: Tokenizer for processing text
        device: Device to run processing on
        config: Optional configuration dictionary

    Returns:
        data_module: Dictionary containing dataset and collator
    """
    # Extract configuration if provided
    data_path = config.get('data_path', None) if config else None
    speech_folder_path = config.get(
        'speech_data_folder', None) if config else None
    wav_config_path = config.get('wav_config_path', None) if config else None
    wav_model_path = config.get('wav_model_path', None) if config else None

    # Create dataset
    train_dataset = SpeechDataset(
        tokenizer=tokenizer,
        device=device,
        data_path=data_path,
        speech_folder_path=speech_folder_path,
        wav_config_path=wav_config_path,
        wav_model_path=wav_model_path
    )

    # Create data collator
    data_collator = SpeechDataCollator(tokenizer=tokenizer)

    return {
        'train_dataset': train_dataset,
        'eval_dataset': None,  # No evaluation dataset for now
        'data_collator': data_collator
    }


def create_dataloader(data_module, batch_size=2, shuffle=True):
    """
    Create a DataLoader from a data module.

    Args:
        data_module: Data module containing dataset and collator
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data

    Returns:
        dataloader: DataLoader instance
    """
    return DataLoader(
        data_module['train_dataset'],
        batch_size=batch_size,
        collate_fn=data_module['data_collator'],
        shuffle=shuffle
    )
