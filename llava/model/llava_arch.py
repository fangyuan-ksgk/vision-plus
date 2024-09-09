import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

class LlavaMetaModel(nn.Module):
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__()
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=config.get("delay_load", False))
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in config.get("mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

    def get_vision_tower(self):
        return self.vision_tower

    def initialize_vision_modules(self, model_args):
        # Update config with vision-related parameters
        self.config.mm_vision_tower = model_args.vision_tower
        self.config.mm_hidden_size = self.vision_resampler.hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.config.mm_patch_merge_type = model_args.mm_patch_merge_type

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

        # Add image_newline parameter if required :: serves as a 'separator' token between image and text (learnable token)
        if "unpad" in self.config.mm_patch_merge_type and not hasattr(self, "image_newline"):
            self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size) / (self.config.hidden_size ** 0.5))

        # Ensure gradients are enabled for mm_projector
        for p in self.mm_projector.parameters():
            p.requires_grad = True

        
        
