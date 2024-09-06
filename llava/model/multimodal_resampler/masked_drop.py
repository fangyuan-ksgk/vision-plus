import torch
import torch.nn as nn

import random 

class MaskedDrop(nn.Moduel):
    def __init__(self, model_args):
        super().__init__()

        self.mode = model_args.mm_mask_drop_mode
        self.skip_percentage = model_args.mm_mask_drop_skip_percentage
        self.ratio = model_args.mm_mask_drop_ratio
        self.ratio_upper = model_args.mm_mask_drop_ratio_upper # randomize ratio of mask drop? 
        self.ratio_lower = model_args.mm_mask_drop_ratio_lower

    def forward(self, image_features, *args, **kwargs):

        if not self.training: # this funny trick is an in-place augmentation used for training only
            return image_features
        
        if self.skip_percentage > random.random(): # don't mask with 'skip_percentage'
            return image_features 

        masked_features = []

        for image_feature in image_features:
            num_tokens = image_feature.shape[0]
            if self.mode == "fixed":
                num_keep = int(num_tokens * self.ratio)
                masked_features.append(self.random_masking(image_feature.unsqueeze(0), num_keep)[0][0])
                raise NotImplementedError("Random masking not implemented yet")
            

    @property
    def config(self):
        return {
            "mm_resampler_type": "masked_drop",
            "mm_mask_drop_mode": self.mode,
            "mm_mask_drop_skip_percentage": self.skip_percentage,
            "mm_mask_drop_ratio": self.ratio,
            "mm_mask_drop_ratio_upper": self.ratio_upper,
            "mm_mask_drop_ratio_lower": self.ratio_lower,
        }
    
    def random_masking(self, x, len_keep): # if you could gather, why not randomize indices and in-place replace with noisey values?
        """ 
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        FY: you want to keep the spatial ordering, so you need to shuffle back after masking right?
        """
        N, L, D = x.shape # batch, length, dim 

        noise = torch.randn(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1) # how to index back to original order

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore