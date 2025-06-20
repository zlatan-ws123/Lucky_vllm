import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
import librosa
IMAGE_TOKEN = "[IMAGE]"

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://kkgithub.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        processor_img,
        processor_audio,
        tokenizer,
        llm_type="minicpm",
        root_path="./data/images/",
        sr=16000,
        query_nums=512,
        add_generation_prompt=False,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.root_path = root_path
        self.tokenizer = tokenizer
        self.processor_img = processor_img
        self.processor_audio = processor_audio
        self.llm_type = llm_type
        self.sr = sr
        self.query_nums = query_nums
        self.add_generation_prompt = add_generation_prompt

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #cust_dir = ["coco", "gqa", "textvqa", "vg", "ocr_vqa"]
        #if "image" in self.raw_data[i] and not any(dir_name in self.raw_data[i]["image"] for dir_name in cust_dir):
        #    self.raw_data[i]["image"] = "coco/train2017/" + self.raw_data[i]["image"]
        if "image" in self.raw_data[i] and os.path.exists(os.path.join(self.root_path, self.raw_data[i]["image"])):
            image = Image.open(os.path.join(self.root_path, self.raw_data[i]["image"])).convert("RGB")
            ''' resize image
            width,height = image.size
            min_edge = min(width, height)
            max_edge = max(width, height)
            if min_edge<224:
                scale = -(-224//min_edge)
                image=image.resize((int(width*scale), int(height*scale)),Image.BILINEAR)
            elif max_edge>1024:
                scale = -(-1024/max_edge)
                if int(width*scale) > 224 and int(height*scale) > 224:
                    image=image.resize((int(width*scale), int(height*scale)),Image.BILINEAR)
            '''
        else:
            image = None
        audio = None
        if "audio" in self.raw_data[i].keys():
            audio, sr = librosa.load(os.path.join(self.root_path, self.raw_data[i]["audio"]), sr=self.sr)
        content_sys = ["You function as an AI assistant, capable of recognizing various types of data, such as images, audio, and text, and can answer questions based on user requests.",
                       "As an AI assistant, you have the ability to interpret multiple modalities of data, including pictures, sounds, and written words, and respond to inquiries as per user needs.",
                       "Your role encompasses identifying diverse data modalities—images, audio, and text—as an AI assistant, and you address questions according to user specifications.",
                       "You serve as an AI assistant with the capacity to discern different forms of data, like visuals, audios, and writings, and you provide answers in line with user queries.",
                       "In your capacity as an AI assistant, you are able to process various modalities of data—images, audio, and text—and you can resolve questions according to user demands.",
                       "You act as an AI assistant, equipped to recognize a range of data types, such as images, audio, and text, and you are designed to answer questions as requested by users.",
                       "It's part of your function as an AI assistant to identify different modalities of data, including images, audio, and text, and to provide answers to user questions.",
                       "You are designed to recognize various forms of data—images, audio, and text—as an AI assistant, and you are there to answer questions as users require.",
                       "One of your key capabilities as an AI assistant is to discern multiple data modalities—images, audio, and text—and to address questions in accordance with user requests.",
                       "You operate as an AI assistant, with the ability to interpret a variety of data types, such as images, audio, and text, and you are tasked with answering questions based on user needs."]
        
        ret = preprocess(
            self.raw_data[i]["conversations"],
            self.tokenizer,
            content_sys[i%len(content_sys)],
            self.processor_img,
            self.processor_audio,
            self.sr,
            image,
            audio,
            #self.raw_data[i]["image"],
            llm_type=self.llm_type,
            add_generation_prompt=self.add_generation_prompt,
        )
        #if image is not None:
        #    img_token_len = [-(-image.size[0]//16) * -(-image.size[1]//16),
        #                     -(-image.size[0]//32) * -(-image.size[1]//32),
        #                     -(-image.size[0]//32) * -(-image.size[1]//32)]
        #else:
        #    img_token_len = [-(-224//16) * -(-224//16),
        #                     -(-224//32) * -(-224//32),
        #                     -(-224//32) * -(-224//32)]
        img_token_len = [729, 729, 729]

        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["target"],
            audio_tgt_sizes=[1,1500],
            attention_mask=torch.ones_like(ret["input_ids"], dtype=torch.bool),
            cross_attention_mask_img=[torch.ones(img_token_len[i], dtype=torch.bool)for i in range(3)],
            cross_attention_mask_audio=torch.ones(self.query_nums, dtype=torch.bool),
            pixel_values=ret["pixel_values"],
            audio=ret["audio"],
            position_ids=ret["position_ids"],
            position_ids_img=ret["position_ids_img"],
        )

        return ret

def data_collator(examples, padding_value=0, max_length=768, gen_collator=False):
    def trim_and_pad(seq, batch_first, padding_value, max_length=max_length):
        return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)

    collator_dict = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "cross_attention_mask_img": [[], [], []],
        "cross_attention_mask_audio": [],
        "audio_tgt_sizes": [],
        "audio": [],
        "pixel_values": [],
        "position_ids": [],
        "position_ids_img": [],
    }
    max_text_num = 0
    max_img_num = [0, 0, 0]
    for example in examples:
        for key in collator_dict.keys():
            if key == "input_ids":
                max_text_num = max(example[key].shape[0], max_text_num)
            elif key == "cross_attention_mask_img":
                for i in range(3):
                    collator_dict[key][i].append(example[key][i])
                    max_img_num[i] = max(example[key][i].shape[0], max_img_num[i])
                continue
            elif key == "position_ids_img":
                collator_dict[key].append(example[key].transpose(0,1))
                continue
            elif key == "position_ids":
                collator_dict[key].append(example[key].transpose(0,1))
                continue
            collator_dict[key].append(example[key])
        #max_img_num = max(example["cross_attention_mask_img"].shape[0], max_img_num)

    collator_dict["input_ids"] = trim_and_pad(
        collator_dict["input_ids"],
        batch_first=True,
        padding_value=padding_value,
        max_length=max_text_num
    ).to(torch.int64)
    collator_dict["position_ids"] = torch.nn.utils.rnn.pad_sequence(
        collator_dict["position_ids"],
        batch_first=True,
        padding_value=padding_value,
    ).to(torch.int64).permute(2,0,1)
    collator_dict["position_ids_img"] = torch.nn.utils.rnn.pad_sequence(
        collator_dict["position_ids_img"],
        batch_first=True,
        padding_value=padding_value,
    ).to(torch.int64).permute(2,0,1)
    if gen_collator:
        collator_dict["labels"] = torch.vstack(collator_dict["labels"])
    else:
        collator_dict["labels"] = trim_and_pad(
            collator_dict["labels"],
            batch_first=True,
            padding_value=-100,
            max_length=max_text_num
        ).to(torch.long)
    collator_dict["attention_mask"] = trim_and_pad(
        collator_dict["attention_mask"],
        batch_first=True,
        padding_value=padding_value,
        max_length=max_text_num
    )
    collator_dict["cross_attention_mask_img"] = [trim_and_pad(
        collator_dict["cross_attention_mask_img"][i],
        batch_first=True,
        padding_value=padding_value,
        max_length=max_img_num[i]
    ) for i in range(3)]
    collator_dict["audio"] = None #torch.vstack(collator_dict["audio"])
    collator_dict["cross_attention_mask_audio"] = torch.vstack(collator_dict["cross_attention_mask_audio"])
    collator_dict["audio_tgt_sizes"] = torch.tensor(collator_dict["audio_tgt_sizes"])
    collator_dict["cross_attention_mask"] = [collator_dict["cross_attention_mask_audio"],
            collator_dict["cross_attention_mask_img"]]
    #del collator_dict["cross_attention_mask_audio"]
    #del collator_dict["cross_attention_mask_img"]

    return collator_dict

def conversation_to_ids(conversation, tokenizer, llm_type=None, cut_max=1536, add_generation_prompt=False):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    assert llm_type in ["lucky", ], "An unsupported LLM type"
    if llm_type == "lucky":
        input_ids, context, raw_msg = conversation_to_ids_lucky(
            conversation, tokenizer, add_generation_prompt
        )
    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))[:cut_max]
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))[:cut_max]

    # build target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            # assert end token(for qwen2)
            if hasattr(tokenizer, "eos_token_id"):
                target[i - 1] = tokenizer.eos_token_id
            else:
                target[i - 1] = tokenizer.eos_token_id

    ######## 2D position ids #####
    position_ids = torch.arange(0, ids.shape[-1]).repeat(2,1)
    img_ids = torch.vstack([torch.arange(0, 9).repeat(9, 1).transpose(0,1).flatten(),
                            torch.arange(0, 9).repeat(9)])
    middle = (81 - 9) // 2
    align = torch.where(torch.from_numpy(np.hstack(input_ids, dtype=np.int32))[:cut_max]==151665)[0][0] + middle
    position_ids[:, torch.where(torch.from_numpy(np.hstack(input_ids, dtype=np.int32))[:cut_max]==151665, True, False)] = img_ids + align
    ##############################

    ######## 2d img position ids for cross attention #######
    h, w = 27, 27
    position_ids_img = torch.vstack([torch.arange(0, h).repeat(w, 1).transpose(0,1).flatten(),
                            torch.arange(0, w).repeat(h)])
    position_ids_img = position_ids_img // 3 + align
    #######################################################
    
    return {
        "input_ids": ids,
        "target": target,
        "raw_msg": raw_msg,
        "position_ids": position_ids,
        "position_ids_img": position_ids_img,
    }


def conversation_to_ids_gen(conversation, tokenizer, llm_type=None, cut_max=1536, add_generation_prompt=True):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    assert llm_type in ["lucky", ], "An unsupported LLM type"
    if llm_type == "lucky":
        input_ids, context, raw_msg = conversation_to_ids_lucky(
            conversation, tokenizer, add_generation_prompt
        )
    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    ids = torch.concat([ids, torch.tensor([151667]+[151668]*576, dtype=torch.int32)])

    ######## 2D position ids #####
    position_ids = torch.arange(0, ids.shape[-1]).repeat(2,1)
    img_ids = torch.vstack([torch.arange(0, 9).repeat(9, 1).transpose(0,1).flatten(),
                            torch.arange(0, 9).repeat(9)])
    img_ids_gen = torch.vstack([torch.arange(0, 24).repeat(24, 1).transpose(0,1).flatten(),
                            torch.arange(0, 24).repeat(24)])
    middle = (81 - 9) // 2
    align = torch.where(ids==151665)[0][0] + middle
    align_gen = torch.where(ids==151667)[0][0] + 1
    position_ids[:, torch.where(ids==151665, True, False)] = img_ids + align
    position_ids[:, torch.where(ids==151668, True, False)] = img_ids_gen + align_gen
    ##############################

    ######## 2d img position ids for cross attention #######
    h, w = 27, 27
    position_ids_img = torch.vstack([torch.arange(0, h).repeat(w, 1).transpose(0,1).flatten(),
                            torch.arange(0, w).repeat(h)])
    position_ids_img = position_ids_img // 3 + align
    #######################################################
    
    return {
        "input_ids": ids,
        "target": None,
        "raw_msg": raw_msg,
        "position_ids": position_ids,
        "position_ids_img": position_ids_img,
    }



def conversation_to_ids_lucky(conversation, tokenizer, add_generation_prompt=False):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=add_generation_prompt,
    )
    input_ids = np.array(input_ids)

    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>"))[0]
    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        st = assistant_idx + 1  # assistant\n
        for eot_idx in eot_idxs:
            if eot_idx > st:
                context[st: eot_idx + 1] = 0
                break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    # print(raw_msg)
    return input_ids, context, raw_msg


def preprocess(
    conversation,
    tokenizer,
    content_sys,
    processor_img,
    processor_audio,
    sr=16000,
    image=None,
    audio=None,
    image_file="",
    llm_type=None,
    Template="llava-med",
    add_generation_prompt=False,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    #assert len(conversation) > 1, "conversation length must large than 2"
    if Template=="llava-med":
        assert conversation[0]["from"] in ["human", "system"], "the first role must be user"
    else:
        assert conversation[0]["role"] in ["user", "system"], "the first role must be user"

        
    img_str = ""
    if image is not None:
        try:
            images = processor_img(images=image, return_tensors="pt", do_resize=True)["pixel_values"]
        except:
            images = torch.ones((1,3,384,384))
            print(image_file)
        img_str = "<image>\n"
    else:
        images = torch.zeros((1,3,384,384))
        #images = torch.zeros((1,3,224,224))
            
    #if audio is not None:
    #    audio = processor_audio(audio, sampling_rate=sr, return_tensors="pt").input_features
    #    conversation[0]["value"] = "[This is an additional audio]\n" + img_str #"[AUDIO]" * 16 + "\n" + img_str
    #else:
    #    audio = None #torch.zeros((1,80,3000))
    audio = None #torch.zeros((1,80,3000))
            
    role_map = {"system": "system", "human": "user", "gpt":"assistant"}
    if Template=="llava-med":
        conversation = [{"role": role_map[message["from"]], "content": message["value"]} for message in conversation]
        if "<image>" in conversation[0]["content"]:
            conversation[0]["content"] = conversation[0]["content"].replace(
                "<image>", "\n"
            )
        #conversation[0]["content"] = f"<|vision_start|>{IMAGE_TOKEN * 81}<|vision_end|>\n" + conversation[0]["content"]

    else:
        conversation = [{"role": role_map[message["from"]], "content": message["value"]} for message in conversation]
        if "<image>" in conversation[0]["content"]:
            conversation[0]["content"] = conversation[0]["content"].replace(
                "<image>", "\n"
            )
        #conversation[0]["content"] = f"<|vision_start|>{IMAGE_TOKEN * 81}<|vision_end|>\n" + conversation[0]["content"]
    if conversation[0]['role'] != "system":
        conversation.insert(0, {'role': 'system', 'content': content_sys})

    #conversation[0]["content"] = f"<|vision_start|>{IMAGE_TOKEN * 81}<|vision_end|>\n" + conversation[0]["content"]
    conversation[0]["content"] = f"{IMAGE_TOKEN * 81}\n" + conversation[0]["content"]
    input_dict = conversation_to_ids(conversation, tokenizer, llm_type, add_generation_prompt=add_generation_prompt)

    input_dict["pixel_values"] = images
    input_dict["audio"] = audio
    return input_dict


def preprocess_gen(
    conversation,
    tokenizer,
    content_sys,
    processor_img,
    processor_audio,
    sr=16000,
    image=None,
    audio=None,
    image_file="",
    llm_type=None,
    Template="llava-med",
    add_generation_prompt=False,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    #assert len(conversation) > 1, "conversation length must large than 2"
    if Template=="llava-med":
        assert conversation[0]["from"] in ["human", "system"], "the first role must be user"
    else:
        assert conversation[0]["role"] in ["user", "system"], "the first role must be user"

        
    img_str = ""
    if image is not None:
        try:
            images = processor_img(images=center_crop_arr(image, 384*2), return_tensors="pt", do_resize=True)["pixel_values"]
        except:
            images = torch.ones((1,3,384,384))
            print(image_file)
        img_str = "<image>\n"
    else:
        images = torch.zeros((1,3,384,384))
        #images = torch.zeros((1,3,224,224))
            
    audio = None #torch.zeros((1,80,3000))
            
    role_map = {"system": "system", "human": "user", "gpt":"assistant"}
    if Template=="llava-med":
        conversation = [{"role": role_map[message["from"]], "content": message["value"]} for message in conversation]
        # conversation[0]["content"] = conversation[0]["content"] + f"{IMAGE_TOKEN * 81}\n"

    else:
        conversation = [{"role": role_map[message["from"]], "content": message["value"]} for message in conversation]
        # conversation[0]["content"] = conversation[0]["content"] + f"{IMAGE_TOKEN * 81}\n"
        
    if conversation[0]['role'] != "system":
        conversation.insert(0, {'role': 'system', 'content': content_sys})

    conversation[0]["content"] = f"{IMAGE_TOKEN * 81}\n" + conversation[0]["content"]
    input_dict = conversation_to_ids_gen(conversation, tokenizer, llm_type, add_generation_prompt=add_generation_prompt)

    input_dict["pixel_values"] = images
    input_dict["audio"] = audio
    return input_dict
