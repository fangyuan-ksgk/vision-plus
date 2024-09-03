from vision_encoder import ViT
from decoder_languge_model import DecoderLanguageModel

class VisionLanguageModel(nn.Module):
    def __init__(self, n_embd, image_embed_dim, vocab_size, n_layer, img_size, patch_size, num_heads, num_blks, emb_dropout, blk_dropout):
        super().__init__()
        num_hiddens = image_embed_dim  # Set num_hiddens equal to image_embed_dim
        assert num_hiddens % num_heads == 0, "num_hiddens must be divisible by num_heads"
        self.vision_encoder = ViT(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)
        self.decoder = DecoderLanguageModel(n_embd, image_embed_dim, vocab_size, num_heads, n_layer, use_images=True)

    def forward(self, img_array, idx, targets=None):
        image_embeds = self.vision_encoder(img_array)

        if image_embeds.nelement() == 0 or image_embeds.shape[1] == 0:
            raise ValueError("somethign is messed up with the ViT model. It's returning an empty tensor or the embedding dimension is empty")

        if targets is not None:
            logits, loss = self.decoder(idx, image_embeds, targets)
            return logits, loss
        else:
            logits = self.decoder(idx, image_embeds)
            return logits

    def generate(self, img_array, idx, max_new_tokens):
      image_embeds = self.vision_encoder(img_array)

      if image_embeds.nelement() == 0 or image_embeds.shape[1] ==0:
        raise ValueError("somethign is messed up with the ViT model. It's returning an empty tensor or the embedding dimension is empty")

      generated_tokens = self.decoder.generate(idx, image_embeds, max_new_tokens)
      return generated_tokens