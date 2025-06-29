import torch
from torch import nn, Tensor
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from torchvision import transforms
import open_clip

__all__ = ["CLIPSVTransformerModel", "CLIPSVOpenclipModel"]


class CLIPSVTransformerModel(nn.Module):

    """
    input an image, return a differentiable loss with respect to the original image
    """
    def __init__(self, clip_name: str, target_img = Tensor, device=torch.device("cuda")):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.target_img = target_img
        self.eval().requires_grad_(False).to(device)
        self.device = device
        self.i_processor = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=None),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        # prepare image embedding
        self.image_embedding = self.calc_img_embedding(target_img).to(device)
        self.criteria = nn.CosineEmbeddingLoss()

    def calc_img_embedding(self, x: Tensor) -> Tensor:
        x = self.i_processor(x)
        vision_outputs = self.clip.vision_model(pixel_values=x.to(self.device))
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds
    
    def forward(self, x: Tensor) -> Tensor:
        embed = self.calc_img_embedding(x).to(self.device)
        # move the two vectors to the same device
        return self.criteria(self.image_embedding.to(embed.device), embed, torch.ones(1).to(embed.device))
    
class CLIPSVOpenclipModel(nn.Module):
    """
    input an image, return a differentiable loss with respect to the original image
    """
    def __init__(self, clip_name: str, target_img = Tensor, device=torch.device("cuda"), resolution=(224, 224)):
        super().__init__()
        clip, _, _ = open_clip.create_model_and_transforms(clip_name)
        self.clip = clip
        self.target_img = target_img
        self.eval().requires_grad_(False).to(device)
        self.device = device
        self.i_processor = transforms.Compose(
            [
                transforms.Resize(resolution, antialias=None),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        # prepare image embedding
        self.image_embedding = self.calc_img_embedding(target_img).to(device)
        self.criteria = nn.CosineEmbeddingLoss()
        print(f"finished initializing the model {clip_name}")

    def calc_img_embedding(self, x: Tensor) -> Tensor:
        x = self.i_processor(x)
        image_embeds = self.clip.encode_image(x.to(self.device))
        # normalization
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

    def forward(self, x: Tensor) -> Tensor:
        embed = self.calc_img_embedding(x).to(self.device)
        return self.criteria(self.image_embedding.to(embed.device), embed, torch.ones(1).to(embed.device))
