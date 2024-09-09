from abc import ABC, abstractmethod

import math
import torch
import torch.nn as nn
from multimodal_encoder.builder import build_vision_tower
from multimodal_resampler.builder import build_vision_resampler
from multimodal_projector.builder import build_vision_projector
from constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

class LlavaMetaModel(nn.Module):
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__()
        
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
            
            
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    
    padding_height = (current_height - original_height) // 2
    padding_width = (current_width - original_width) // 2

    height_start = padding_height
    width_start = padding_width
    
    # Slice the tensor
    unpadded_tensor = tensor[:, 
                             height_start : current_height - height_start,
                             width_start : current_width - width_start]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    
    @abstractmethod
    def get_model(self):
        pass 
    
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def add_token_per_frame(self, image_feature):
        """ 
        Image_feature: [num_frames, num_patches, feature_dim]
        Assume square grid (grid is an ensemble of patches, so H=W=sqrt(num_patches))
        We append an 'image-end' token to each frame
        Detail: shape of image_newline: [feature_dim] -- so we operate after the multi-modal projector
        """
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        expand_tokens = self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
        image_feature = torch.cat((image_feature, expand_tokens), dim=-1) # [feature_dim, num_frames, num_patches+1]
        image_feature = image_feature.permute(1, 2, 0).contiguous() # [num_frames, num_patches+1, feature_dim]
        return image_feature 
    
    def encode_images(self, images): # Should work for list of images here (!)
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = [self.add_token_per_frame(image_feature) for image_feature in image_features]
        return image_features
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images):
        """ 
        Logic: 
        Process images into embedding vectors, then slot them into input_embedding sequence together with the text embeddings, with padding on the maximal sequence length
        Quite importantly, the input_ids need to specify where the image is in the sequence by using IMAGE_TOKEN_INDEX value
        Questions: 
        - Where does the input_ids gets processed, meaning the insertion of IMAGE_TOKEN_INDEX values
        - 2 Values are not used here: position_ids, past_key_values, why we need them?
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Encode images | original implementation concatenate all images along row axis, and split the resulting features | should be included into .encode_images function really
        image_features = self.encode_images(images)

        # Process input ids and labels
        input_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels = [labs[mask] for labs, mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # Somehow all the IMAGE TOKEN are set to be a specific value | Embedding is also done through the vision encoder
            if num_images == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            # Split input ids and labels at image tokens
            split_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_chunks = []
            cur_labels_chunks = []
            for i in range(len(split_indices)-1):
                if split_indices[i+1] - split_indices[i] > 1:
                    cur_input_ids_chunks.append(cur_input_ids[split_indices[i]+1:split_indices[i+1]])
                    cur_labels_chunks.append(labels[batch_idx][split_indices[i]+1:split_indices[i+1]])

            # Interleave text embeddings and image features
            cur_input_embeds = []
            cur_labels = []
            for i, (ids_chunk, labels_chunk) in enumerate(zip(cur_input_ids_chunks, cur_labels_chunks)):
                cur_input_embeds.append(self.get_model().embed_tokens(ids_chunk))
                cur_labels.append(labels_chunk)
                if i < num_images:
                    cur_input_embeds.append(image_features[batch_idx])
                    cur_labels.append(torch.full((image_features[batch_idx].shape[0],), IGNORE_INDEX, device=labels_chunk.device, dtype=labels_chunk.dtype))

            new_input_embeds.append(torch.cat(cur_input_embeds))
            new_labels.append(torch.cat(cur_labels))

        # Pad sequences to max length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels[0].device)

        for i, (cur_embeds, cur_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_embeds.shape[0]
            new_input_embeds_padded.append(torch.cat((cur_embeds, torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device))))
            new_labels_padded[i, :cur_len] = cur_labels
            attention_mask[i, :cur_len] = True

        new_input_embeds = torch.stack(new_input_embeds_padded) # [batch_size, max_len, feature_dim]

        return None, None, attention_mask, past_key_values, new_input_embeds, new_labels_padded
        
          