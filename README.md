# In-Context Former: Lightning-fast Compressing Context for Large Language Model
This repository is the official implementation of the paper "In-Context Former: Lightning-fast Compressing Context for Large Language Model" (Findings of EMNLP 2024).

## Requirements

To get started, please clone this repository and install packages as:

```bash
git clone https://github.com/wonderful9462/IC-Former.git
conda create -n icformer python=3.9
pip install -r requirements.txt
```

## Dataset

- For pretraining, we use a part of [Pile](https://github.com/EleutherAI/the-pile) Dataset, and you can download it [here](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).

- For instruction fine-tuning, we use [Prompt-with-Context](https://huggingface.co/datasets/sggetao/PwC) Dataset

## Run

You can simply train IC-Former by running:

```bash
# Pretrain an IC-Former
bash pretrain.sh
# Instruction fine-tune IC-Former
bash finetune.sh
# Generate response on the test set
bash generate.sh
```

## Arguments Explanation

`pretrain.py`, `finetune.py` and `generate.py` share the same command line arguments. Explanation is as follows:

- `data_path`: Path to the jsonl format data for training or testing.
- `lm_path`: Path to the directory of target language model.
- `icformer_path`: Path to the directory of IC-Former checkpoint. If you specify this argument, IC-Former will initialize from the checkpoint, or IC-Former will randomly initialize.
- `save_path`: Path to save the checkpoints, training logs and generated responses. Default set to "./output"
- `num_hidden_layers`: When you random initialize IC-Former, you can specify the layer number of IC-Former.
- `num_query_tokens`: When you random initialize IC-Former, you can specify the digest tokens number of IC-Former.
- `max_seq_len`: The maximum length of the IC-Former input. Default set to 512.
- `use_chunk`: To specify whether to split input into chunks when its length exceeds the capacity of digest embeddings.
- `max_chunk_len`: The maximum length of single chunk, this argument can determine how many chunks will the input be split into.
- `seed`: Random seed to ensure the reproducibility of the experiment.
- `shuffle`: Whether to shuffle the dataset.
- `lr`: learning rate of the optimizer. Default set to 1e-4.
- `max_epoch`: Total training epoch. Default set to 1.
- `gradient_accumulation`: Since we only implemented the input of a single sample, in order to achieve the same result as the batch input, we designed a gradient manager to accurately simulate the result of the batch input. So you can think of this parameter as "traditional batch_size".
- `avg_level`: An argument to tell gradient manager how to average the gradient during gradient accumulation. You can choose 'sentence' or 'token' to macro or micro average the gradient separately.
- `save_interval`: To indicate how often to save a checkpoint. The number represents the amount of input context.
- `save_optimizer`: Whether to save optimizer states when save a checkpoint
- `clip_grad`: Whether to clip the gradient of IC-Former parameters
- `max_norm`: If you specify to clip gradient, this argument will determine the maximum norm of parameter gradient.
- `max_new_tokens`: The maximum tokens to generate when you run `generate.py`. Default set to 256.
- `output_file`: The file name to save  the generated responses. Default set to "output.jsonl".

## Citation

If you find this code useful, please kindly cite our work as:

```bibtex
@article{wang2024context,
  title={In-Context Former: Lightning-fast Compressing Context for Large Language Model},
  author={Wang, Xiangfeng and Chen, Zaiyi and Xie, Zheyong and Xu, Tong and He, Yongyi and Chen, Enhong},
  journal={arXiv preprint arXiv:2406.13618},
  year={2024}
}
```

