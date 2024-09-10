import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, rank0_print


import os
from transformers import AutoTokenizer
import torch
from peft import PeftModel

from language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base=None, load_lora=False, device_map="auto", attn_implementation="flash_attention_2"):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load LlavaLlamaForCausalLM model
    llava_cfg = LlavaConfig.from_pretrained(model_path)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path if model_base is None else model_base,
        config=llava_cfg,
        torch_dtype=torch.float32  # Full precision
    )
    
    # Load LoRA if specified
    if load_lora and model_base is not None:
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    
    # Add special tokens for multimodal processing
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    
    # Determine context length
    context_len = getattr(model.config, 'max_position_embeddings', 2048)
    
    return tokenizer, model, image_processor, context_len


def initialize_llava_model(
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    intermediate_size=11008,
    max_position_embeddings=2048,
    vision_model_name="openai/clip-vit-large-patch14",
    vision_tower_config=None,
    mm_vision_select_layer=-2,
    mm_use_im_start_end=True,
    mm_use_im_patch_token=True
):
    # Initialize LlavaConfig
    llava_config = LlavaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        vision_model_name=vision_model_name,
        vision_tower_config=vision_tower_config,
        mm_vision_select_layer=mm_vision_select_layer,
        mm_use_im_start_end=mm_use_im_start_end,
        mm_use_im_patch_token=mm_use_im_patch_token
    )

    # Initialize model
    model = LlavaLlamaForCausalLM(config=llava_config)

    # Initialize tokenizer (you might want to use a specific tokenizer or create a new one)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)

    # Add special tokens for multimodal processing
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor, max_position_embeddings