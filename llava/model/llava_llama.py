from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from torch.nn import CrossEntropyLoss

from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from multimodal_encoder.builder import build_vision_tower
from multimodal_resampler.builder import build_vision_resampler
from multimodal_projector.builder import build_vision_projector

from llava_arch import LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    vision_tower: Optional[str] = None

    model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    mm_vision_tower: str = "openai/clip-vit-base-patch32"
    mm_vision_select_feature: str = "patch"
    mm_vision_select_layer: int = -1
    mm_hidden_size: int = 768  # this one match the pre-trained vision encoder
    use_im_start_end: bool = False
    use_im_patch_token: bool = True
    delay_load: bool = True
    mm_resampler_type: Optional[str] = None
    hidden_size: int = 768  # This one should match with the language model
    # rope_scaling: Optional[dict] = {}


def prep_llava_llama_tokenizer(model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '<|begin_of_text|>',
        'eos_token': '<|eot_id|>',
        'additional_special_tokens': [
                                    '<|start_header_id|>', 
                                    '<|end_header_id|>',
                                    '<|start_image_id|>',
                                    '<|end_image_id|>',
                                    '<|start_video_id|>',
                                    '<|end_video_id|>',
                                    ]
    })
    return tokenizer

class LlavaLlamaModel(LlamaModel): # Remove LlavaMetaModel to reduce the trouble
    
    # Might need to resize token embeddings here ... (TBD)
    
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        
        self.tokenizer = prep_llava_llama_tokenizer()
        self.resize_token_embeddings(len(self.tokenizer))
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=getattr(config, "delay_load", False))
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

    def get_vision_tower(self):
        return self.vision_tower

    def initialize_vision_modules(self, model_args):
        # Update config with vision-related parameters
        self.config.mm_vision_tower = model_args.mm_vision_tower
        self.config.mm_hidden_size = self.vision_resampler.mm_hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature

        # Load vision tower if not already loaded
        if self.vision_tower is None:
            self.vision_tower = build_vision_tower(model_args)
            self.vision_resampler = build_vision_resampler(model_args, vision_tower=self.vision_tower)
        else:
            self.vision_tower.load_model()

        # Ensure gradients are enabled for vision resampler
        for p in self.vision_resampler.parameters():
            p.requires_grad = True

        # Initialize mm_projector if not already present
        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config, vision_cfg=self.vision_tower.config)

        # Ensure gradients are enabled for mm_projector
        for p in self.mm_projector.parameters():
            p.requires_grad = True


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    
    def __init__(self, config):
        LlamaForCausalLM.__init__(self, config)
        
        config.model_type = "llava_llama"
        
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initalize weight and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model 

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """ 
        Intuition: 
        1. We explicitly preprocess sequence input embeddings instead of using input_ids, given the extra modality
        2. For 'dpo' or 'sft', the propagation differs in whether we calculate the CELoss within the forward function.
        - for DPO, we don't calculate CELoss by not including 'labels' in the forward function computation
        - for SFT, we propagate LlamaModel forward function with 'labels' included
        So DPO's loss is not CELoss (simPO has CELoss as one of its component, but calculating them outside the forward function should be the norm)
        """

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        modalities: Optional[List[str]] = ["image"],
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        if images is not None:
            (input_ids, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images, modalities)
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """ 
        Malfunctioning .... or at least incomplete preparation function ...
        I guess this function also assumes input_ids are already interleaving text & image tokens
        Then it forms a dictionary with: input_ids, attention_mask, images, image_sizes, etc. 
        Note that vision encoding is by no way completed with this function ....
        """
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
        

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)