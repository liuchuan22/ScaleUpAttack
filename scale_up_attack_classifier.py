from data import get_NIPS17_loader
from attacks import (
    BIM,
    FGSM,
    PGD,
    MI_RandomWeight,
    MI_FGSM,
    MI_CosineSimilarityEncourager,
    MI_SAM,
    MI_CommonWeakness,
    SSA_CommonWeakness,
    MI_SVRE,
    VMI_Outer_CommonWeakness,
    VMI_Inner_CommonWeakness,
    DI_MI_FGSM,
    MI_RAP
)
from models import *
import torch
from utils import get_list_image, save_image, get_image
import os
from tqdm import tqdm
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
import argparse
import random

parser = argparse.ArgumentParser(description='scale_up_adv')
parser.add_argument('--snum', type=int, default='64', help='number of surrogate models trained on ImageNet')
parser.add_argument('--target_path', type=str, default='./resources/cifar10/tabby_cat.jpg', help='target image path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = args.target_path

target_image = get_image(image_path)
resizer = transforms.Resize((224, 224), antialias=None)
target_image = resizer(target_image).unsqueeze(0).to(device)
snum = args.snum
snum_list = random.sample(range(64), snum)

train_models = []

origin_test_models = [
    alexnet,
    densenet121,
    densenet161,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_v2_l,
    googlenet,
    inception_v3,
    maxvit_t,
    mnasnet0_75,
    mnasnet1_3,
    mobilenet_v3_small,
    mobilenet_v3_large,
    regnet_x_400mf,
    regnet_y_400mf,
    resnet18,
    resnet50,
    resnet101,
    resnet152,
    shufflenet_v2_x0_5,
    squeezenet1_0,
    vgg13_bn,
    vgg16,
    vit_b_32,
    vit_b_16,
    vit_l_16,
    vit_l_32,
    wide_resnet101_2,
    wide_resnet50_2,
    densenet169,
    efficientnet_b4,
    efficientnet_v2_m, 
    efficientnet_v2_s,
    mnasnet0_5,
    mnasnet1_0,
    mobilenet_v2,
    regnet_x_3_2gf,
    regnet_x_800mf,
    regnet_x_8gf,
    regnet_y_1_6gf,
    regnet_y_32gf,
    regnet_y_800mf,
    resnet34,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
    squeezenet1_1,
    vgg11,
    vgg11_bn,
    vgg13,
    vgg16_bn,
    vgg19,
]

# 12 models from huggingface
origin_hf_models = [
    "microsoft/beit-large-patch16-512",
    "google/vit-base-patch16-224",
    "hf_hub:timm/ghostnet_100.in1k",
    "facebook/deit-base-distilled-patch16-224",
    "hf_hub:timm/lcnet_050.ra2_in1k",
    "hf_hub:timm/repvgg_a2.rvgg_in1k",
    "hf_hub:timm/dpn98.mx_in1k",
    "hf_hub:timm/twins_svt_large.in1k",
    "hf_hub:timm/nfnet_l0.ra2_in1k",
    "nvidia/mit-b0",
    "google/bit-50",
    "microsoft/cvt-13",
]

for i, model_1 in enumerate(origin_test_models):
    if i in snum_list:
        now_model = ModelWrapper(model_1(pretrained=True), use_transform=True).to(device)
        now_model.eval()
        now_model.requires_grad_(False)
        train_models.append(now_model)
        snum_list.remove(i)

for i, model_2 in enumerate(origin_hf_models):
    if (i+52) in snum_list:
        snum_list.remove(i+52)
        if "timm" in model_2:
            now_model = HFTimmModel(model_2, target_image)
            now_model.requires_grad_(False)
            train_models.append(now_model)
        else:
            now_model = HFTransformerModel(model_2, target_image)
            now_model.requires_grad_(False)
            train_models.append(now_model)

assert len(snum_list) == 0

print("total surrogate models: #", len(train_models))

loader = get_NIPS17_loader(batch_size=1)

# call a standard resnet101 model for the target ground truth.
rn101 = BaseNormModel(resnet101(pretrained = True)).to(device)
rn101.eval()
rn101.requires_grad_(False)
with torch.no_grad():
    logits = rn101.forward(target_image)
print(logits.argmax(dim=-1))

class CELoss_Targeted:
    def __init__(self):
        self.count = 0
        self.logit = logits.argmax(dim=-1).to(device)
        self.criteria = nn.CrossEntropyLoss()
        self.repeated_logit = logits.argmax(dim=-1).to(device).repeat(8)

    def __call__(self, input_logit, *args):
        self.count += 1
        self.count %= 100
        assert input_logit.dim() == 2 and input_logit.shape[1] == 1000, "logit not in ImageNet1k"
        if input_logit.shape[0] == 8:
            loss = self.criteria(input_logit, self.repeated_logit)
        else:
            loss = self.criteria(input_logit, self.logit)
        loss *= -1
        if self.count == 0:
            print(loss)
        return loss

# An example class for untargeted attack. We try to maximize the loss between image and the original image class.
class CELoss_Untargeted:
    def __init__(self):
        self.count = 0
        self.criteria = nn.CrossEntropyLoss()
    def __call__(self, x, y):
        self.count += 1
        self.count %= 100
        loss = self.criteria(x,y) 
        if self.count == 0:
            print(loss)
        return loss

attacker = SSA_CommonWeakness(train_models,
                            epsilon= 8 / 255,
                            # epsilon = 2/255, # This is for untargeted setting.
                            step_size=1 / 255,
                            # reverse_start_step=10, # For specific algorithms, e.g. VMI 
                            total_step=40, 
                            criterion=CELoss_Targeted())
print(attacker.__class__)

from pathlib import Path
base_path = Path('./results/classifiers/targeted')
final_path = base_path / f"{attacker.total_step}_step_{repr(attacker)}_{snum}_snum_{int(attacker.epsilon*255)}_eps"
for subdir in [Path('./results'), './results/classifiers', base_path, final_path, final_path / "adv", final_path / "perturb"]:
    Path(subdir).mkdir(parents=True, exist_ok=True)

for i, (x, y) in enumerate(tqdm(loader)):
    if i >= 100:
        break
    x = x.cuda()
    adv_x = attacker(x.clone(), y)
    save_image(adv_x, os.path.join(final_path, f"adv/{i}.png"))
    perturb = ((adv_x - x) + attacker.epsilon) / (2 * attacker.epsilon)
    print(torch.max(perturb), torch.min(perturb))
    save_image(perturb, os.path.join(final_path, f"perturb/{i}_perturb.png"))
