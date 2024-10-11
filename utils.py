import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel

"""
Loss function
"""

def kl_div(input, target, reduction='batchmean'):
    KLDivLoss = torch.nn.KLDivLoss(reduction=reduction, log_target=True)
    log_input = F.log_softmax(input, dim=-1)
    log_target = F.log_softmax(target, dim=-1)
    return KLDivLoss(log_input, log_target)

"""
Tools
"""

def parse_args():
    parser = argparse.ArgumentParser()
    # path args
    parser.add_argument('--data_path', type=str, default='./data/pile_small.jsonl')
    parser.add_argument('--lm_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--icformer_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='./output')
    # model args
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_query_tokens', type=int, default=128)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--use_chunk', action='store_true')
    parser.add_argument('--max_chunk_len', type=int, default=512)
    # parser.add_argument('--encode', action='store_true')
    # training args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--save_optimizer', action='store_true')
    parser.add_argument('--save_scheduler', action='store_true')
    parser.add_argument('--gradient_accumulation', type=int, default=32)
    parser.add_argument('--clip_grad', action='store_true')
    parser.add_argument('--max_norm', type=float, default=2.0)
    parser.add_argument('--avg_level', type=str, default="token", choices=["token", "sentence"])
    # generation args
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--output_file', type=str, default="output.jsonl")

    args = parser.parse_args()
    return args

def current_date_time(delta_hours=0):
    now = datetime.datetime.now() + datetime.timedelta(hours=delta_hours)
    formatted_date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    return formatted_date_time

def get_nb_trainable_parameters(model: torch.nn.Module):
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def print_trainable_parameters(model: torch.nn.Module):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_nearest_tokens(model:PreTrainedModel, soft_prompt):
    vocab = model.get_input_embeddings().weight.data
    nearest_ids = (soft_prompt[0] @ vocab.T).argmax(-1)
    return nearest_ids


"""
Visualize the loss curve
"""

r"""
use example:

>>> from utils import Visualizer
>>> visualizer = Visualizer("./output/checkpoint-1000")
>>> visualizer.show(save_path="/PATH/TO/SAVE", dpi=150)
"""

class Visualizer:
    def __init__(self, checkpoint_dir):
        log_file = os.path.join(checkpoint_dir, "trainer_state.json")
        self.log = json.load(open(log_file, "r"))
        self.global_steps, self.loss_record = [], []

        steps_per_epoch = self.log["steps_per_epoch"]
        for log in self.log["log_history"]:
            self.global_steps.append(log["epoch"] * steps_per_epoch + log["step"])
            self.loss_record.append(log["loss"])
    
    def show(self, save_path=None, dpi=200, linewidth=0.5, **kwargs):
        plt.figure(figsize=(12,8), dpi=dpi)
        plt.xlabel("Global Steps")
        plt.ylabel("Loss")
        plt.yticks(range(0,10,1))
        plt.plot(self.global_steps, self.loss_record, linewidth=linewidth, **kwargs)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)
