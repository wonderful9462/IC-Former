import os
import gc
import math
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from peft import PeftModel
from utils import current_date_time
from icformer import ICFormerModel

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

class Trainer:
    def __init__(
        self,
        model, 
        dataset, 
        optimizer,
        scheduler=None,
        max_epoch=1,
        save_interval=1000,
        save_dir='./output',
        save_optimizer=False,
        save_scheduler=False,
        avg_level='token',
        gradient_accumulation=1,
        shuffle=True,
        clip_grad=True,
        max_norm=2,
    ):
        """
        A pytorch-lightning style trainer for training models.
        'save_optimizer' is a boolean value to save the optimizer state dict.
        'avg_level' is a string value to indicate how to average loss during gradient accumulation. You can choose 'token' or 'sentence'.
        """
        self.max_epoch = max_epoch
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.model = model
        self.dataset = dataset
        self.steps_per_epoch = len(dataset)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation = gradient_accumulation
        self.max_norm = max_norm
        self.shuffle = shuffle
        self.clip_grad = clip_grad

        self.handler = GradHandler(avg_level=avg_level)
        self.log_record = []
        self.log_file = os.path.join(self.save_dir, f"{current_date_time()}.log")

        os.makedirs(self.save_dir, exist_ok=True)

    def train(
        self,
        start_step=0,
        start_epoch=0,
    ):
        tqdm.write(f"Start training at {current_date_time()}.")
        for epoch in range(start_epoch, self.max_epoch):
            self.model.zero_grad()
            torch.cuda.empty_cache()
            
            if self.shuffle:
                self.dataset.shuffle()

            with tqdm(total=self.steps_per_epoch-start_step, ncols=100, unit='B') as pbar:
                for step in range(start_step, self.steps_per_epoch):
                    data = self.dataset[step]
                    loss = self.model.train_step(step, data)
                    self.handler.append(loss)
                    self.handler.backward(loss)
                    pbar.update(1)

                    if (step+1) % self.gradient_accumulation == 0 or (step+1) == self.steps_per_epoch:
                        self.handler.apply_grad(self.optimizer)
                        self.log(
                            epoch=epoch, 
                            step=step+1, 
                            loss=self.handler.compute_loss(), 
                            grad=self.handler.compute_grad_norm(self.optimizer),
                        )
                        self.optimize()
                        self.handler.clear()

                        if self.scheduler is not None:
                            self.scheduler.step(step)

                    if (step+1) % self.save_interval == 0:
                        self.save(epoch, step+1)

                if (step+1) % self.save_interval != 0:
                    self.save(epoch, step+1)
                start_step = 0
    
    def log(self, epoch, step, loss, grad, **kwargs):
        record = {}
        if loss is not None:
            record["loss"] = loss
        if grad is not None:
            record["grad_norm"] = grad
        record["learning_rate"] = self.optimizer.param_groups[0]['lr']
        record["epoch"] = epoch
        record["step"] = step
        record.update(kwargs)
        tqdm.write(f"{record}")
        self.log_record.append(record)

    def optimize(self):
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(parameters=self.optimizer.param_groups[0]['params'], max_norm=self.max_norm, norm_type=2)
            # torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value=0.005)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save(self, epoch, step):
        module_to_save = self.model.get_pretrained_model()
        save_directory = os.path.join(self.save_dir, f"checkpoint-{self.steps_per_epoch*epoch+step}")
        module_to_save.save_pretrained(save_directory=save_directory)
        param = {}
        if hasattr(self.model, 'digest_embeddings'):
            param['digest_embeddings'] = self.model.digest_embeddings
        if hasattr(self.model, 'memory_embeddings'):
            param['memory_embeddings'] = self.model.memory_embeddings
        if hasattr(self.model, 'AE'):
            param['AE'] = self.model.AE
        if hasattr(self.model, 'LM'):
            param['LM'] = self.model.LM
        if hasattr(self.model, 'FT'):
            param['FT'] = self.model.FT
        
        torch.save(param, os.path.join(save_directory, "param.pt"))
        
        if self.save_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.pt"))

        if self.save_scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(save_directory, "scheduler.pt"))

        trainer_state = {"steps_per_epoch": self.steps_per_epoch, "log_history": self.log_record}
        json.dump(trainer_state, open(os.path.join(save_directory, "trainer_state.json"), "w"), indent=2)

class GradHandler:
    """
    Gradient handler is designed for handling the gradient accumulation, loss averaging and gradient norm calculation.
    The handler recieves the loss of every token and accumulates them to compute the average loss.
    """
    def __init__(self, avg_level='token'):
        self.loss_list = []
        self.total_len = 0
        self.avg_level = avg_level

    def append(self, loss:torch.Tensor=None):
        if loss is not None or len(loss) > 0:
            if self.avg_level == 'token':
                self.total_len += len(loss)
                self.loss_list.append(loss.sum().item())
            elif self.avg_level == 'sentence':
                self.total_len += 1
                self.loss_list.append(loss.mean().item())

    def backward(self, loss:torch.Tensor=None):
        if loss is not None or len(loss) > 0:
            if self.avg_level == 'token':
                loss.sum().backward()
            elif self.avg_level == 'sentence':
                loss.mean().backward()

    def compute_grad_norm(self, optimizer:torch.optim.Optimizer):
        grad_norm = []
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    grad_norm.append(param.grad.detach().norm())
        all_norm = torch.tensor(grad_norm).norm().item()
        return all_norm
    
    def compute_loss(self):
        if self.total_len == 0:
            return 0
        return sum(self.loss_list) / self.total_len
    
    def apply_grad(self, optimizer:torch.optim.Optimizer):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad and param.grad is not None:
                    grad = param.grad / self.total_len
                    param.grad = grad
            
    def clear(self):
        self.total_len = 0
        self.loss_list.clear()
        gc.collect()


class BaseModel(nn.Module):
    """
    A base class model suitable for training with Trainer, where all models inheriting from this class should at least implement 
    the train_step and get_pretrained_model methods.
    """
    def __init__(
        self,
        language_model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer,
    ):
        super().__init__()
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.prepare_tokenizer(self.tokenizer)

    @staticmethod
    def prepare_tokenizer(tokenizer:PreTrainedTokenizer):
        tokenizer.pad_token_id = 0
        tokenizer.add_bos_token = True
    
    @torch.no_grad()
    def generate(
        self,
        inputs_embeds:torch.Tensor,
        max_new_tokens:int=256,
        skip_special_tokens:bool=True,
        streaming:bool=False,
        return_output:bool=True,
        **kwargs,
    ):
        # Greedy decoding
        inputs_embeds = inputs_embeds.to(dtype=self.language_model.dtype)
        past_key_values = None
        output_ids = []
        for _ in range(max_new_tokens):
            output = self.language_model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
            next_token_id = output.logits[0][-1].argmax(dim=-1)
            output_ids.append(next_token_id)
            if streaming:
                response = self.tokenizer.decode(output_ids)
                if not response: pass
                elif response[-1] == "\n": print()
                print(response.split('\n')[-1], end='\r', flush=True)
            if next_token_id == self.tokenizer.eos_token_id:
                break
            past_key_values = output.past_key_values
            next_embeds = self.language_model.get_input_embeddings()(torch.tensor([[next_token_id]], device=self.language_model.device))
            inputs_embeds = next_embeds
        if return_output:
            outputs = self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens, **kwargs)
            outputs = outputs.strip()
            return outputs, output_ids
        return None

    @torch.no_grad()
    def convert_ids_to_embeds(self, input_ids):
        if isinstance(input_ids, list):
            if isinstance(input_ids[0], list):
                input_ids = torch.tensor(input_ids)
            else:
                input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(device=self.language_model.device, dtype=torch.long)
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        return embeddings
    
    def get_backbone(self):
        if isinstance(self.language_model, PeftModel):
            return self.language_model.model.model
        return self.language_model.model
    
    def get_pretrained_model(self):
        raise NotImplementedError("get_pretrained_model method is not implemented.")
    
    def train_step(self, step, data):
        raise NotImplementedError("train_step method is not implemented.")
    
    def get_soft_prompt(self):
        raise NotImplementedError("get_soft_prompt method is not implemented.")


class ICFormer(BaseModel):
    def __init__(
        self, 
        icformer:ICFormerModel, 
        language_model:PreTrainedModel, 
        tokenizer:PreTrainedTokenizer
    ):
        super().__init__(language_model=language_model, tokenizer=tokenizer)
        self.icformer = icformer

        self.digest_embeddings = nn.Parameter(torch.zeros([1, icformer.config.num_query_tokens, icformer.config.hidden_size], device=icformer.device, dtype=icformer.dtype))
        self.AE = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))
        # self.LM = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))

        self.max_seq_len = 512
        self.max_chunk_len = 512
        self.use_chunk = True
        self.encode = False

    def train_step(self, step, data):
        context = data
        context_ids = self.tokenizer(context)['input_ids']
        if len(context_ids) > self.max_seq_len: # random split
            last_start = len(context_ids) - self.max_seq_len
            start = random.randint(0, last_start)
            context_ids = context_ids[start:start+self.max_seq_len]
        label_ids = context_ids + [self.tokenizer.eos_token_id]
        context_embeds = self.convert_ids_to_embeds(context_ids)
        # Whether to use the language model to encode the context.
        if self.encode:
            with torch.no_grad():
                context_embeds = self.language_model.model(inputs_embeds=context_embeds)[0]

        label_embeds = self.convert_ids_to_embeds(label_ids)
        soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
        inputs_embeds = torch.cat([soft_prompt, self.AE, label_embeds], dim=1)
        # shifted right
        logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-len(label_ids)-1:-1]
        ae_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
        return ae_loss
    
    def get_soft_prompt(
        self, 
        query_embeds=None, 
        input_ids=None, 
        inputs_embeds=None, 
        use_chunk=False,
        **kwargs,
    ):
        """
        Implement the soft prompt generation method.
        'use_chunk' is a boolean value to specify whether to apply divide-and-conquer strategy.
        """
        if query_embeds is None:
            query_embeds = self.digest_embeddings

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if input_ids is not None:
            inputs_embeds = self.convert_ids_to_embeds(input_ids)
        embeddings = inputs_embeds.to(device=self.icformer.device)

        # Causal attention mask for query tokens.
        query_mask = torch.tril(torch.ones([1, query_embeds.shape[1], query_embeds.shape[1]]))
        
        if not use_chunk:
            cross_attn_mask = torch.ones([1, query_embeds.shape[1], embeddings.shape[1]])
            attention_mask = torch.cat([cross_attn_mask, query_mask], dim=-1).to(device=self.icformer.device)
            hidden_states = torch.cat([embeddings, query_embeds], dim=1)
            soft_prompt = self.icformer(
                query_embeds=query_embeds,
                context_hidden_states=hidden_states,
                context_attention_mask=attention_mask,
                **kwargs,
            )[0].to(device=self.language_model.device)
        else:
            soft_prompt = []
            chunk_num = math.ceil(embeddings.shape[1] / self.max_chunk_len)
            chunk_size = math.ceil(embeddings.shape[1] / chunk_num)
            
            for index in range(chunk_num):
                chunk_embeds = embeddings[:,index*chunk_size:(index+1)*chunk_size]
                chunk_mask = torch.ones([1, query_embeds.shape[1], chunk_embeds.shape[1]])
                attention_mask = torch.cat([chunk_mask, query_mask], dim=-1).to(device=self.icformer.device)
                hidden_states = torch.cat([chunk_embeds, query_embeds], dim=1).to(device=self.icformer.device)
                chunk_soft_prompt = self.icformer(
                    query_embeds=query_embeds,
                    context_hidden_states=hidden_states,
                    context_attention_mask=attention_mask,
                    **kwargs,
                )[0][:,-self.digest_embeddings.shape[1]:]
                soft_prompt.append(chunk_soft_prompt.to(device=self.language_model.device))
            soft_prompt = torch.cat(soft_prompt, dim=1)
        return soft_prompt

    def get_pretrained_model(self):
        return self.icformer


class ICFormerQA(ICFormer):
    def __init__(
        self, 
        icformer:ICFormerModel, 
        language_model:PreTrainedModel, 
        tokenizer:PreTrainedTokenizer
    ):
        super().__init__(icformer=icformer, language_model=language_model, tokenizer=tokenizer)
        self.AE.requires_grad = False
        # self.LM.requires_grad = False
        self.FT = nn.Parameter(torch.zeros([1, 1, icformer.config.hidden_size], device=language_model.device, dtype=language_model.dtype))

        self.pre_embeds = self.convert_ids_to_embeds(self.tokenizer("<s>[INST] Response the Prompt based on the below text:\n\n")['input_ids'])
        self.post_embeds = self.convert_ids_to_embeds(self.tokenizer("[/INST]")['input_ids'])

        self.alpha = 1.0 
        self.max_label_len = 65535

    def train_step(self, step, data):
        entropy_loss, kl_loss = 0, 0
        context, prompt, label = data

        label_ids = self.tokenizer(label)['input_ids'][:self.max_label_len] + [self.tokenizer.eos_token_id]
        context_ids = self.tokenizer(context)['input_ids'][:self.max_seq_len]
        prompt_ids = self.tokenizer(prompt)['input_ids']
        label_len = len(label_ids)

        label_embeds = self.convert_ids_to_embeds(label_ids)
        context_embeds = self.convert_ids_to_embeds(context_ids)
        prompt_embeds = self.convert_ids_to_embeds(prompt_ids)

        if self.encode:
            with torch.no_grad():
                context_embeds = self.language_model.model(inputs_embeds=context_embeds)[0]
        
        soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
        inputs_embeds = torch.cat([self.pre_embeds, soft_prompt, self.FT, prompt_embeds, self.post_embeds, label_embeds], dim=1)
        logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-label_len-1:-1]

        if self.alpha > 0:
            entropy_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
        if self.alpha < 1:
            with torch.no_grad():
                inputs_embeds = torch.cat([self.pre_embeds, context_embeds, prompt_embeds, self.post_embeds, label_embeds], dim=1)
                target_logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-label_len-1:-1]
            kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(target_logits, dim=-1), reduction="none").sum(-1)
        loss = self.alpha * entropy_loss + (1-self.alpha) * kl_loss
        return loss
    
    @staticmethod
    def prepare_tokenizer(tokenizer:PreTrainedTokenizer):
        tokenizer.pad_token_id = 0
        # The bos token has been added in the context.
        tokenizer.add_bos_token = False

# Reproduction of the ICAE model.

# class ICAE(BaseModel):
#     def __init__(self, language_model:PreTrainedModel, tokenizer:PreTrainedTokenizer):
#         super().__init__(language_model=language_model, tokenizer=tokenizer)

#         self.memory_embeddings = nn.Parameter(torch.zeros([1, 128, 4096], device=language_model.device, dtype=language_model.dtype))
#         self.AE = nn.Parameter(torch.zeros([1, 1, 4096], device=language_model.device, dtype=language_model.dtype))
#         self.LM = nn.Parameter(torch.zeros([1, 1, 4096], device=language_model.device, dtype=language_model.dtype))
        
#         self.max_seq_len = 512
#         self.max_chunk_size = 512
#         self.lm_ratio = 0.4
#         self.reserve_len = 8
#         self.min_tokens_for_lm = 64
#         self.use_chunk = False
    
#     def get_soft_prompt(
#         self, 
#         inputs_embeds=None, 
#         attention_mask=None, 
#         use_chunk=False, 
#         **kwargs
#     ):
#         if attention_mask is not None:
#             attention_mask = attention_mask.to(device=self.language_model.device)
#         embeddings = inputs_embeds.to(device=self.language_model.device)
#         if use_chunk:
#             soft_prompt = []
#             chunk_num = math.ceil(embeddings.shape[1] / self.max_chunk_size)
#             chunk_size = math.ceil(embeddings.shape[1] / chunk_num)

#             for index in range(chunk_num):
#                 chunk_embeds = embeddings[:,index*chunk_size:(index+1)*chunk_size]
#                 hidden_states = torch.cat([chunk_embeds, self.memory_embeddings], dim=1).to(device=self.language_model.device)
#                 chunk_soft_prompt = self.get_backbone()(
#                     inputs_embeds=hidden_states,
#                     attention_mask=attention_mask,
#                     **kwargs,
#                 )[0][:,-self.memory_embeddings.shape[1]:]
#                 soft_prompt.append(chunk_soft_prompt)
#             soft_prompt = torch.cat(soft_prompt, dim=1)
#         else:
#             embeddings = torch.cat([embeddings, self.memory_embeddings], dim=1)
#             soft_prompt = self.get_backbone()(
#                 inputs_embeds=embeddings,
#                 attention_mask=attention_mask,
#                 **kwargs,
#             )[0][:, -self.memory_embeddings.shape[1]:]
#         return soft_prompt.to(device=self.language_model.device)
    
#     @torch.no_grad()
#     def generate(
#         self,
#         inputs_embeds:torch.Tensor,
#         max_new_tokens:int=256,
#         skip_special_tokens:bool=True,
#         streaming:bool=True,
#         return_output:bool=True,
#         **kwargs,
#     ):
#         with self.language_model.disable_adapter():
#             return super().generate(
#                 inputs_embeds=inputs_embeds, 
#                 max_new_tokens=max_new_tokens, 
#                 skip_special_tokens=skip_special_tokens, 
#                 streaming=streaming, 
#                 return_output=return_output, 
#                 **kwargs
#             )
    
#     def train_step(self, step, data):
#         lm = random.random() < self.lm_ratio
#         context = data
#         context_ids = self.tokenizer(context)['input_ids']
#         if lm: # lm loss
#             if len(context_ids) > self.max_seq_len: # random split
#                 last_start = len(context_ids) - self.max_seq_len
#                 start = random.randint(0, last_start)
#                 left_ids, right_ids = context_ids[start:start+self.max_seq_len], context_ids[start+self.max_seq_len:]
#             else:
#                 pivot = random.randint(0, len(context_ids)-1)
#                 left_ids, right_ids = context_ids[:pivot+1], context_ids[pivot+1:]
#             if len(right_ids) >= self.min_tokens_for_lm:
#                 right_ids = right_ids[:self.max_seq_len]
#                 left, right = self.convert_ids_to_embeds(left_ids), self.convert_ids_to_embeds(right_ids)
#                 label_ids = torch.tensor(right_ids[self.reserve_len:], device=self.language_model.device)
#                 soft_prompt = self.get_soft_prompt(inputs_embeds=left, use_chunk=self.use_chunk)
#                 inputs_embeds = torch.cat([soft_prompt, self.LM, right], dim=1)
#                 with self.language_model.disable_adapter():
#                     logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-len(label_ids)-1:-1]
#                 lm_loss = F.cross_entropy(logits, label_ids, reduction="none")
#                 return lm_loss
#             else:
#                 lm = False
#         if not lm: # ae loss
#             if len(context_ids) > self.max_seq_len: # random split
#                 last_start = len(context_ids) - self.max_seq_len
#                 start = random.randint(0, last_start)
#                 context_ids = context_ids[start:start+self.max_seq_len]
#             label_ids = context_ids + [self.tokenizer.eos_token_id]
#             context_embeds = self.convert_ids_to_embeds(context_ids)
#             label_embeds = self.convert_ids_to_embeds(label_ids)
#             soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
#             inputs_embeds = torch.cat([soft_prompt, self.AE, label_embeds], dim=1)
#             with self.language_model.disable_adapter():
#                 logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-len(label_ids)-1:-1]
#             ae_loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
#             return ae_loss
    
#     def get_pretrained_model(self):
#         return self.language_model


# class ICAEQA(ICAE):
#     def __init__(self, language_model:PreTrainedModel, tokenizer:PreTrainedTokenizer):
#         super().__init__(language_model=language_model, tokenizer=tokenizer)
#         self.AE.requires_grad = False
#         self.LM.requires_grad = False
#         self.FT = nn.Parameter(torch.zeros([1, 1, 4096], device=language_model.device, dtype=language_model.dtype))

#         self.pre_embeds = self.convert_ids_to_embeds(self.tokenizer("<s>[INST] Response the Prompt based on the below text:\n\n")['input_ids'])
#         self.post_embeds = self.convert_ids_to_embeds(self.tokenizer("[/INST]")['input_ids'])

#         self.max_label_len = 65535

#     def train_step(self, step, data):
#         context, prompt, label = data

#         label_ids = self.tokenizer(label)['input_ids'][:self.max_label_len] + [self.tokenizer.eos_token_id]
#         context_ids = self.tokenizer(context)['input_ids']
#         prompt_ids = self.tokenizer(prompt)['input_ids']
#         label_len = len(label_ids)

#         label_embeds = self.convert_ids_to_embeds(label_ids)
#         context_embeds = self.convert_ids_to_embeds(context_ids)
#         prompt_embeds = self.convert_ids_to_embeds(prompt_ids)
        
#         soft_prompt = self.get_soft_prompt(inputs_embeds=context_embeds, use_chunk=self.use_chunk)
#         inputs_embeds = torch.cat([self.pre_embeds, soft_prompt, self.FT, prompt_embeds, self.post_embeds, label_embeds], dim=1)

#         with self.language_model.disable_adapter():
#             logits = self.language_model(inputs_embeds=inputs_embeds).logits[0][-label_len-1:-1]

#         loss = F.cross_entropy(logits, torch.tensor(label_ids, device=logits.device), reduction="none")
#         return loss
    
#     @staticmethod
#     def prepare_tokenizer(tokenizer:PreTrainedTokenizer):
#         tokenizer.pad_token_id = 0
#         tokenizer.add_bos_token = False
