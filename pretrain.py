import os
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

from icformer import (
    ICFormerConfig, 
    ICFormerModel, 
)
from modules import Trainer, ICFormer
from data_utils import PileDataset
from utils import parse_args, seed_everything

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    data = PileDataset(args.data_path)
    tokenizer = LlamaTokenizer.from_pretrained(args.lm_path, use_fast=True)

    if args.icformer_path: # Load icformer checkpoint
        icformer = ICFormerModel.from_pretrained(args.icformer_path, device_map='cuda', torch_dtype=torch.bfloat16)
    else:                  # Random initialize icformer
        icformer_config = ICFormerConfig()
        icformer_config.num_hidden_layers = args.num_hidden_layers
        icformer_config.num_query_tokens = args.num_query_tokens
        icformer = ICFormerModel(icformer_config).to(dtype=torch.bfloat16, device='cuda')

    language_model = AutoModelForCausalLM.from_pretrained(args.lm_path, device_map='cuda', torch_dtype=torch.bfloat16)
    language_model.requires_grad_(False)

    model = ICFormer(icformer, language_model, tokenizer)
    model.max_seq_len = args.max_seq_len
    model.max_chunk_len = args.max_chunk_len
    model.use_chunk = args.use_chunk
    # model.encode = args.encode
    
    if args.icformer_path: # Load digest embeddings and special tokens embeddings
        ckpt = torch.load(os.path.join(args.icformer_path, 'param.pt'))
        with torch.no_grad():
            model.digest_embeddings.copy_(ckpt['digest_embeddings'])
            model.AE.copy_(ckpt['AE'])
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if os.path.exists(os.path.join(args.icformer_path, 'optimizer.pt')):
        ckpt = torch.load(os.path.join(args.icformer_path, 'optimizer.pt'))
        optimizer.load_state_dict(ckpt)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    if os.path.exists(os.path.join(args.icformer_path, 'scheduler.pt')):
        ckpt = torch.load(os.path.join(args.icformer_path, 'scheduler.pt'))
        scheduler.load_state_dict(ckpt)

    trainer = Trainer(
        model=model,
        dataset=data,
        optimizer=optimizer,
        scheduler=None, # You can create your own scheduler
        max_epoch=args.max_epoch,
        save_interval=args.save_interval,
        save_dir=args.save_path,
        save_optimizer=args.save_optimizer,
        save_scheduler=args.save_scheduler,
        avg_level=args.avg_level,
        gradient_accumulation=args.gradient_accumulation,
        shuffle=args.shuffle,
        clip_grad=args.clip_grad,
        max_norm=args.max_norm,
    )

    trainer.train(
        start_epoch=0, # specify the last epoch if you want to resume training
        start_step=0,  # specify the last step(assuming dataset is not shuffled) if you want to resume training
    )
