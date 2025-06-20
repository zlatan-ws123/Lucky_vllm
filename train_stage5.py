#!/usr/bin/env python
# coding: utf-8
import gc
from data_process_lucky import SupervisedDataset, data_collator, preprocess
from modeling_lucky_fixed import LuckyForCausalLM, LuckyConfig
import torch
import json
from typing import Union
from tqdm import tqdm
from functools import partial

from transformers.utils import logging
from transformers import LlamaConfig, PretrainedConfig
from transformers import AutoTokenizer
from transformers import AutoModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from utils import LinearDecayLR
from c_adamw import AdamW as C_AdamW
from transformers import AutoProcessor
from transformers import AutoImageProcessor, SwinForImageClassification
from transformers import WhisperProcessor
from transformers import TrainingArguments, Trainer
from torch.amp import autocast, GradScaler
import os
import copy
import random
import math
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, interleave_datasets
from PIL import Image
#new_alloc = torch.cuda.memory.CUDAPluggableAllocator('./auto_alloc.so','mem_alloc','mem_free');
#torch.cuda.memory.change_current_allocator(new_alloc)
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.cpu(), alpha=1 - rate)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['_adapter', '_tower', 'proj_addi', 'gate', 'diffloss']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(name)#[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#torch.set_num_threads(1)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
logger = logging.get_logger(__name__)
config = LuckyConfig()

model = LuckyForCausalLM.from_pretrained("/mnt/model/Lucky2_5_stage3_epoch1", attn_implementation="sdpa", ignore_mismatched_sizes=True)

model.to(torch.bfloat16)
#checkpoint = torch.load("/mnt/model/checkpoint_Lucky2_5_stage2_008_.pth", map_location="cpu")
#model.load_state_dict(checkpoint['model_state_dict'], strict=False)
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.to(torch.bfloat16)
checkpoint = torch.load("/mnt/model/checkpoint_Lucky2_5_stage5_004_ema.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.train()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}!")
model.to(device)

with open('/home/luca/LLM/data/llava_v1_5_mix665k_audio.json', 'r', encoding='utf-8') as file:
    train_json = json.load(file)[:624000]

print(len(train_json))
random.seed(42)
random.shuffle(train_json)
random.shuffle(train_json)
random.shuffle(train_json)

dataset_cls = SupervisedDataset
tokenizer = AutoTokenizer.from_pretrained(
        "/media/model/Lucky_2_5", trust_remote_code=True
    )

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
processor_img = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

train_dataset = dataset_cls(
        train_json[1024*200:],
        processor_img,
        processor_audio=processor,
        tokenizer=tokenizer,
        llm_type="lucky",
        root_path="/home/luca/LLM/data",
)
#print(train_dataset[0])

batch_size = 1
learning_rate = 5e-5
steps = 1024
tot_updates = int(700000//(batch_size*steps))
training_params = model._get_train_param(stage=5)
#ema_params = [copy.deepcopy(param.detach().cpu()) for param in training_params]
ema_params = checkpoint['ema_params']
dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False,collate_fn=data_collator)
optimizer = C_AdamW(training_params, lr=learning_rate, weight_decay=0)
scheduler = LinearDecayLR(
        optimizer,
        warmup_updates=60,#int(tot_updates*0.02),
        tot_updates=tot_updates,
        lr=learning_rate,
        end_lr=1e-6,)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
print(scheduler.state_dict())
all_loss = 0
n_loss = 0
loss_list = []

def get_batch_samples(epoch_iterator,# epoch_iterator_text,
                      num_batches, batch=1, use_cap=False):
    batch_samples = []
    num_items_in_batch = None
    for i in range(num_batches):
        try:
            batch_samples += [next(epoch_iterator)]
        except StopIteration:
            break

    if len(batch_samples) > 0 and "labels" in batch_samples[0]:
        # For now we don't support object detection
        try:
            num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        except (TypeError, AttributeError):
            pass

    return batch_samples, num_items_in_batch

epoch_iterator = iter(dataloader)
total_updates = 200 #math.ceil(len(train_dataset) / (batch_size * steps))

for n in tqdm(range(total_updates)):
    batch_samples, num_items_in_batch = get_batch_samples(epoch_iterator, steps, batch=batch_size)
    all_loss = 0
    for i in batch_samples:
        inputs = model.prepare_inputs_for_generation(i["input_ids"].to(torch.int64).to(device),
                                                     attention_mask=i["attention_mask"], data = i, position_ids_=i["position_ids"])
        inputs["labels"] = i["labels"].to(torch.long)
        inputs["use_cache"] = False
        try:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                loss = model(**inputs).loss/num_items_in_batch
            if not torch.any(torch.isnan(loss)):
                all_loss += loss.item()
                loss.backward()
            else:
                print("nan case")
                del loss
        except torch.OutOfMemoryError as e:
            print("out of mem!")
            if 'loss' in locals():
                del loss
        #del inputs
    print(f"iter:{n+1} loss:{all_loss}")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    update_ema(ema_params, training_params, rate=0.9999)
    torch.cuda.empty_cache()

torch.save({
   'model_state_dict': model.state_dict(),
   'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
   'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
   'ema_params': ema_params,
}, f"/mnt/model/checkpoint_Lucky2_5_stage5_005_ema.pth")
print("model saved!")
