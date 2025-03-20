"""
This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import torch
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from transformers import T5ForConditionalGeneration, AutoTokenizer

# Import configuration
from configs.train_config import config as default_config

# Import model components
from src.model import GPTConfig, GPT

# Import data loading utilities
from src.data import create_data_module, create_dataloader, get_batch

# Import utility functions 
from src.utils import (
    setup_environment,
    get_lr,
    setup_wandb,
    save_checkpoint,
    smart_tokenizer_and_embedding_resize,
    estimate_loss
)


def parse_args():
    """Parse command line arguments and update config."""
    parser = argparse.ArgumentParser(description='Training script with command-line config options')
    
    # System settings
    parser.add_argument('--device', type=str, help='Device to use for training (e.g., cuda:0)')
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32'], help='Data type for training')
    parser.add_argument('--backend', type=str, help='Backend for distributed training')
    
    # Model architecture
    parser.add_argument('--n_layer', type=int, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, help='Context size for attention')
    parser.add_argument('--dropout', type=float, help='Dropout probability')
    parser.add_argument('--bias', type=bool, help='Whether to use bias in LayerNorm and Linear layers')
    
    # Model paths
    parser.add_argument('--encoder_model_path', type=str, help='Path to encoder model')
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer')
    parser.add_argument('--checkpoint_filename', type=str, help='Checkpoint filename')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--wav_config_path', type=str, help='Path to wav tokenizer config')
    parser.add_argument('--wav_model_path', type=str, help='Path to wav tokenizer model')
    
    # Training control
    parser.add_argument('--is_train', type=bool, help='Whether to train the model')
    parser.add_argument('--out_dir', type=str, help='Output directory for checkpoints')
    parser.add_argument('--eval_interval', type=int, help='Evaluation interval in iterations')
    parser.add_argument('--log_interval', type=int, help='Logging interval in iterations')
    parser.add_argument('--eval_iters', type=int, help='Number of iterations for evaluation')
    parser.add_argument('--eval_only', type=bool, help='Whether to only evaluate the model')
    parser.add_argument('--always_save_checkpoint', type=bool, help='Whether to always save checkpoints')
    parser.add_argument('--init_from', type=str, choices=['scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], 
                        help='Initialize model from scratch, resume, or pretrained GPT2')
    parser.add_argument('--compile', type=bool, help='Whether to compile the model')
    
    # Logging
    parser.add_argument('--wandb_log', type=bool, help='Whether to log to wandb')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, help='Wandb run name')
    
    # Dataset
    parser.add_argument('--dataset', type=str, help='Dataset name')
    
    # Training hyperparameters
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--beta1', type=float, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, help='Beta2 for Adam optimizer')
    parser.add_argument('--grad_clip', type=float, help='Gradient clipping value')
    
    # Learning rate schedule
    parser.add_argument('--decay_lr', type=bool, help='Whether to decay learning rate')
    parser.add_argument('--warmup_iters', type=int, help='Number of warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, help='Number of iterations for learning rate decay')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    
    # Parse and return args
    args = parser.parse_args()
    return args


def update_config_with_args(config, args):
    """Update config with command line arguments."""
    # Convert argparse Namespace to dictionary, filtering out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Update config with args_dict
    config.update(args_dict)
    
    return config


def initialize_model(config, device):
    """Initialize model from scratch, checkpoint or pretrained weights."""
    # Set up model arguments from config
    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'],
        bias=config['bias'], 
        vocab_size=None, 
        dropout=config['dropout']
    )
    
    meta_vocab_size = 4096  # default vocab size
    iter_num = 0  # default starting iteration
    
    if config['init_from'] == 'scratch':
        # Initialize new model from scratch
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif config['init_from'] == 'resume':
        # Resume from a checkpoint
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], config['checkpoint_filename'])
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        # Ensure essential config attributes match the checkpoint
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # Create model with loaded config
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict from checkpoint
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        # Get iteration number from checkpoint
        iter_num = checkpoint['iter_num']
        
    elif config['init_from'].startswith('gpt2'):
        # Initialize from OpenAI GPT-2 weights
        print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
        override_args = dict(dropout=config['dropout'])
        model = GPT.from_pretrained(config['init_from'], override_args)
        
        # Update model args from the initialized model
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    
    # Move model to the specified device
    model.to(device)
    
    # Crop block size if needed
    if config['block_size'] < model.config.block_size:
        model.crop_block_size(config['block_size'])
        model_args['block_size'] = config['block_size']
    
    return model, model_args, iter_num


def setup_tokenizer_and_encoder(config, device):
    """Set up tokenizer and encoder model."""
    print(f"Loading encoder model from {config['encoder_model_path']}")
    
    # Load encoder model
    llm_model = T5ForConditionalGeneration.from_pretrained(config['encoder_model_path'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    
    # Add special tokens to tokenizer and resize model embeddings
    print("Adding special tokens to tokenizer")
    special_tokens = [
        dict(pad_token="[PAD]"),
        dict(pad_token="EOS"),
    ]
    
    for tokens in special_tokens:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=tokens,
            tokenizer=tokenizer,
            model=llm_model,
        )
    
    # Use only the encoder's embedding layer
    print("Extracting encoder's embedding layer")
    llm_model = llm_model.encoder.embed_tokens.to(device)
    
    return tokenizer, llm_model


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Get default config and update with command line arguments
    config = update_config_with_args(default_config.copy(), args)
    
    # Set up training environment
    print("Setting up training environment")
    env = setup_environment(config)
    device = env['device']
    device_type = env['device_type']
    ctx = env['ctx']
    master_process = env['master_process']
    ddp = env['ddp']
    ddp_local_rank = env['ddp_local_rank']
    
    # Initialize model
    print("Initializing model")
    context_length = int(config['block_size'])
    model, model_args, iter_num = initialize_model(config, device)
    
    # Initialize optimizer and gradient scaler
    print("Configuring optimizer")
    optimizer = model.configure_optimizers(
        config['weight_decay'], 
        config['learning_rate'], 
        (config['beta1'], config['beta2']), 
        device_type
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    
    # Compile model if enabled (requires PyTorch 2.0+)
    if config['compile']:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Wrap model in DDP if distributed training
    if ddp:
        print(f"Setting up Distributed Data Parallel (rank {ddp_local_rank})")
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Set up tokenizer and encoder model
    print("Setting up tokenizer and encoder")
    tokenizer, llm_model = setup_tokenizer_and_encoder(config, device)
    
    # Create data loader
    print("Creating data module and dataloader")
    data_module = create_data_module(tokenizer, device, config)
    train_dataloader = create_dataloader(
        data_module,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Initialize wandb logging
    wandb = setup_wandb(config, master_process)
    
    # Get initial batch
    print("Fetching initial batch")
    X, Y = get_batch(train_dataloader,context_length,device, device_type, llm_model)
    
    # Training loop setup
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed
    running_mfu = -1.0
    
    print(f"Beginning training from iteration {iter_num}")
    
    # Main training loop
    while True:
        # Set learning rate for current iteration
        lr = get_lr(config, iter_num) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate and save checkpoint at specified intervals
        if iter_num % config['eval_interval'] == 0 and master_process:
            losses = estimate_loss(model, train_dataloader,context_length,config['eval_iters'], ctx, device, device_type, llm_model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}")
            
            # Log to wandb if enabled
            if wandb:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "lr": lr,
                    "mfu": running_mfu*100,  # convert to percentage
                })
            
            # Save checkpoint
            save_checkpoint(config, model, optimizer, iter_num, model_args)
            
        # Exit if only evaluating
        if iter_num == 0 and config['eval_only']:
            print("Evaluation complete, exiting")
            break
        
        # Forward and backward passes with gradient accumulation
        for micro_step in range(env['gradient_accumulation_steps']):
            # Handle gradient synchronization for DDP
            if ddp:
                model.require_backward_grad_sync = (micro_step == env['gradient_accumulation_steps'] - 1)
            
            # Forward pass with automatic mixed precision
            with ctx:
                try:
                    logits, loss, _ = model(X, Y)
                    loss = loss / env['gradient_accumulation_steps']  # Scale for gradient accumulation
                except Exception as e:
                    print(f"Error in forward pass: {str(e)}")
                    continue
            
            # Prefetch next batch during forward pass
            X, Y = get_batch(train_dataloader,context_length,device, device_type, llm_model)
            
            # Backward pass with gradient scaling if using fp16
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        # Optimizer step and gradient reset
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config['log_interval'] == 0 and master_process:
            lossf = loss.item() * env['gradient_accumulation_steps']
            if local_iter_num >= 5:  # Let training settle before measuring MFU
                mfu = raw_model.estimate_mfu(config['batch_size'] * env['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1
        
        # Check termination condition
        if iter_num > config['max_iters']:
            print(f"Reached maximum iterations ({config['max_iters']}), training complete")
            break
    
    # Clean up DDP resources
    if ddp:
        destroy_process_group()
    
    print("Training finished successfully")


if __name__ == "__main__":
    main()
