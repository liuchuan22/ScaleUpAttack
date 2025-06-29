from data import get_NIPS17_loader
from attacks import SSA_CommonWeakness
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

# add args
parser = argparse.ArgumentParser(description='scale_up_adv')
parser.add_argument('--cnum', type=int, default='12', help='number of surrogate models that has a clip structure')
parser.add_argument('--target_path', type=str, default='./resources/cifar10/tabby_cat.jpg', help='target image path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = args.target_path

target_image = get_image(image_path)
resizer = transforms.Resize((224, 224), antialias=None)
target_image = resizer(target_image).unsqueeze(0).to(device)

cnum = args.cnum
cnum_list = random.sample(range(12), cnum)

train_models = []

clips = [
    "openai/clip-vit-large-patch14",
    "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",
    "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
]

# select the models that are in the cnum_list
clips = [clips[i] for i in cnum_list]
for clip in clips:
    if "openai" in clip:
        clip_model = CLIPSVTransformerModel(clip, target_image, device="cuda")
    elif "timm" in clip:
        clip_model = CLIPSVOpenclipModel(clip, target_image, device="cuda", resolution=(384, 384))
    elif "bigG" in clip:
        clip_model = CLIPSVOpenclipModel(clip, target_image, device="cuda")
    else:
        clip_model = CLIPSVOpenclipModel(clip, target_image, device="cuda")
    train_models.append(clip_model)

print("total surrogate models: #", len(train_models))

loader = get_NIPS17_loader(batch_size=1)

class LossPrinter:
    def __init__(self):
        self.count = 0
    def __call__(self, loss, *args):
        self.count += 1
        self.count %= 100
        loss *= -1
        if self.count >= 0:
            print(loss)
        return loss

attacker = SSA_CommonWeakness(train_models,
                            epsilon= 16 / 255,
                            step_size=1 / 255,
                            total_step=40, criterion=LossPrinter())
print(attacker.__class__)

from pathlib import Path
base_path = Path('./results/clips')
final_path = base_path / f"{attacker.total_step}_step_{repr(attacker)}_{cnum}_cnum_{int(attacker.epsilon*255)}_eps"
for subdir in [Path('./results'), './results/clips', base_path, final_path, final_path / "adv", final_path / "perturb"]:
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
