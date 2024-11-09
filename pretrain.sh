CUDA_VISIBLE_DEVICES=0 python pretrain.py \
--data_path ./data/pile_small.jsonl \
--lm_path meta-llama/Llama-2-7b-hf \
--save_path ./output \
--gradient_accumulation 32 \
--max_seq_len 512 \
--save_optimizer \
--clip_grad \
--shuffle
