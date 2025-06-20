#!/usr/bin/env python
# coding: utf-8
import gc
from data_process_lucky import SupervisedDataset, data_collator
from modeling_lucky_fixed import LuckyForCausalLM, LuckyConfig
import torch
import json
from typing import Union
from tqdm import tqdm

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
#new_alloc = torch.cuda.memory.CUDAPluggableAllocator('./auto_alloc.so','mem_alloc','mem_free');
#torch.cuda.memory.change_current_allocator(new_alloc)
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#torch.set_num_threads(1)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
logger = logging.get_logger(__name__)
config = LuckyConfig()
#model = LuckyForCausalLM.from_pretrained("/media/model/Lucky_2_5_epoch1_lora", attn_implementation="sdpa", ignore_mismatched_sizes=True)
model = LuckyForCausalLM.from_pretrained("/media/model/Lucky_2_5_stage1_new_mid_gate_", attn_implementation="eager", ignore_mismatched_sizes=True)

#vision_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
#vision_model.vision_model.encoder.layers = vision_model.vision_model.encoder.layers[:-1]
#vision_model.vision_model.head = torch.nn.Identity()
#vision_model.requires_grad_(False)
#model.vision_tower = vision_model.vision_model
#model.resize_token_embeddings(151936)
#print(model)
#model.lm_head.weight = copy.deepcopy(model.model.embed_tokens.weight)

#for v_layer in range(6, 27, 5):
#    model.model.layers[v_layer].self_attn.k_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.k_proj.weight)
#    model.model.layers[v_layer].self_attn.k_proj_addi.bias = copy.deepcopy(model.model.layers[v_layer].self_attn.k_proj.bias)
#    model.model.layers[v_layer].self_attn.v_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.v_proj.weight)
#    model.model.layers[v_layer].self_attn.v_proj_addi.bias = copy.deepcopy(model.model.layers[v_layer].self_attn.v_proj.bias)
#    model.model.layers[v_layer].self_attn.o_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.o_proj.weight)
#
#for v_layer in range(5, 27, 5):
#    model.model.layers[v_layer].self_attn.k_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.k_proj.weight)
#    model.model.layers[v_layer].self_attn.k_proj_addi.bias = copy.deepcopy(model.model.layers[v_layer].self_attn.k_proj.bias)
#    model.model.layers[v_layer].self_attn.v_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.v_proj.weight)
#    model.model.layers[v_layer].self_attn.v_proj_addi.bias = copy.deepcopy(model.model.layers[v_layer].self_attn.v_proj.bias)
#    model.model.layers[v_layer].self_attn.o_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.o_proj.weight)
#for v_layer in range(1, 24, 4):
#    model.model.layers[v_layer].self_attn.q_proj_addi.weight = copy.deepcopy(model.model.layers[v_layer].self_attn.q_proj.weight)
#    model.model.layers[v_layer].self_attn.q_proj_addi.bias = copy.deepcopy(model.model.layers[v_layer].self_attn.q_proj.bias)

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(dataset1)
        self.length2 = len(dataset2)

    def __len__(self):
        return self.length1 + self.length2

    def __getitem__(self, idx):
        if idx < self.length1:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.length1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"training on {device}!")
model.to(device)

with open('/media/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json', 'r', encoding='utf-8') as file:
    train_json = json.load(file)
#print(len(train_json))
random.seed(42)
random.shuffle(train_json)
random.shuffle(train_json)
random.shuffle(train_json)
#random.shuffle(train_json)

with open('/mnt/LLM/data/TextOCR/TextOCR-GPT4o_text.json', 'r', encoding='utf-8') as file:
    train_json_ocr = json.load(file)

dataset_cls = SupervisedDataset
tokenizer = AutoTokenizer.from_pretrained(
        "/media/model/Lucky_2_5", trust_remote_code=True
    )

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
processor_img = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

train_dataset = dataset_cls(
        train_json[1024*100:],
        processor_img,
        processor_audio=processor,
        tokenizer=tokenizer,
        llm_type="lucky",
        root_path="/media/data/LLaVA-Pretrain",
)

train_dataset_ocr = dataset_cls(
        train_json_ocr,
        processor_img,
        processor_audio=processor,
        tokenizer=tokenizer,
        llm_type="lucky",
        root_path="/mnt/LLM/data",
)

ds_conb = CombinedDataset(train_dataset, train_dataset_ocr)
batch_size = 2
learning_rate = 5e-4
steps = 256
tot_updates = int(558000//(batch_size*steps))
dataloader = DataLoader(ds_conb,batch_size=batch_size, shuffle=True,collate_fn=data_collator)
#optimizer = torch.optim.AdamW(model._get_train_param(), lr=learning_rate, weight_decay=0)
optimizer = C_AdamW(model._get_train_param(), lr=learning_rate, weight_decay=0)
#optimizer = torch.optim.SGD(model._get_train_param(), lr=learning_rate, momentum=0.9)
scaler = GradScaler()
scheduler = LinearDecayLR(
        optimizer,
        warmup_updates=10,#int(tot_updates*0.02),
        tot_updates=tot_updates,
        lr=learning_rate,
        end_lr=5e-5,)

all_loss = 0
n_loss = 0
loss_list = []

def get_batch_samples(epoch_iterator, num_batches):
    batch_samples = []
    num_items_in_batch = None
    for _ in range(num_batches):
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
total_updates = math.ceil(len(ds_conb) / (batch_size * steps))
for n in tqdm(range(total_updates)):
    batch_samples, num_items_in_batch = get_batch_samples(epoch_iterator, steps)
    all_loss = 0
    for i in batch_samples:
        inputs = model.prepare_inputs_for_generation(i["input_ids"].to(torch.int64).to(device), attention_mask=i["attention_mask"], data = i, position_ids_=i["position_ids"])
        inputs["labels"] = i["labels"].to(torch.long)
        inputs["use_cache"] = False
        try:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                loss = model(**inputs).loss/num_items_in_batch
            all_loss += loss.item()
            loss.backward()
            #scaler.scale(loss).backward()
            del loss
        except torch.OutOfMemoryError as e:
            print("out of mem!")
            if 'loss' in locals():
                del loss
        del inputs
        torch.cuda.empty_cache()
    print(f"iter:{n+1} loss:{all_loss}")
    optimizer.step()
    #scaler.step(optimizer)
    scheduler.step()
    #scaler.update()
    optimizer.zero_grad()
    #if (n+1)%1000==0:
    #    model.merge_and_unload().save_pretrained(f'/media/model/checkpoint_img_audio_{n+1}_lora_pos')
    #    torch.save({
    #        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    #        'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
    #    }, f"/media/model/checkpoint_img_audio_{n+1}_lora.pth")
    #    print("model saved!")

model.save_pretrained(f'/media/model/Lucky_2_5_stage1_new_mid_gate_ocr_fixed')
