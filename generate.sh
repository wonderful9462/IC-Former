CUDA_VISIBLE_DEVICES=0 python generate.py \
--data_path ./data/PwC_test.jsonl \
--lm_path meta-llama/Llama-2-7b-hf \
--icformer_path ./output/checkpoint-239655 \
--save_path ./results \
--max_new_tokens 256 \
--max_seq_ken 1024 \
--max_chunk_len 512 \
--output_file PwC_output.jsonl \
--use_chunk
