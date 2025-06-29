import torch
from torch import nn, Tensor
from torchvision import transforms as trf
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm
from timm.data import create_transform, resolve_model_data_config

class HFTransformerModel(nn.Module):
    def __init__(self, name: str, target_img: Tensor):
        super(HFTransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name)
        self.model = self.model.eval().to(self.device)
        self.model.requires_grad_(False)

        self.target_logit = self.get_logit(target_img)

    def get_logit(self, x: Tensor) -> Tensor:
        # x: Tensor [B,C,H,W] with values in [0,1] assumed
        with torch.no_grad():
            inputs = self.processor(images=x, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
        return logits

    def forward(self, x: Tensor):
        return self.get_logit(x)


class HFTimmModel(nn.Module):
    def __init__(self, name: str, target_img: Tensor):
        super(HFTimmModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = timm.create_model(name, pretrained=True)
        model = model.eval().to(self.device)
        model.requires_grad_(False)
        self.model = model
        data_config = resolve_model_data_config(model)
        self.transforms = create_transform(**data_config, is_training=False)

        self.target_logit = self.get_logit(target_img)

    def get_logit(self, x: Tensor) -> Tensor:
        # x: Tensor [B,C,H,W] with values in [0,1] assumed
        with torch.no_grad():
            x_trans = self.transforms(x)
            x_trans = x_trans.to(self.device)
            output = self.model(x_trans)
        return output

    def forward(self, x: Tensor):
        return self.get_logit(x)
