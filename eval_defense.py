from data import get_NIPS17_loader
from models import *
import torch
from defenses import (
    Randomization,
    JPEGCompression,
    BitDepthReduction,
    NeuralRepresentationPurifier,
    randomized_smoothing_resnet50,
)
from defenses.HighLevelGuidedDenoiser import get_HGD_model
from defenses.RandomizedSmoothing.RSModels import randomized_smoothing_resnet50
from defenses.PurificationDefenses.DiffPure import DiffusionPure
from defenses.PurificationDefenses.DiffPure.sampler import DiffusionSde, DDIM
from torchvision import transforms
from models.unets.guided_diffusion import get_guided_diffusion_unet

from utils import get_list_image, get_image
from typing import List, Callable
import os
import json

test_models = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_50 = BaseNormModel(resnet50(pretrained=True))

origin_test_models = [
    adv_inception_v3,
    ens_adv_inception_resnet_v2,
]

for model in origin_test_models:
    now_model = BaseNormModel(model(pretrained=True)).to(device)
    now_model.eval()
    now_model.requires_grad_(False)
    test_models.append(now_model)

unet = get_guided_diffusion_unet()
diffpure = DiffusionPure(
    sampler=DDIM(unet=unet, img_shape=(3, 256, 256)),
    model=resnet_50,
    pre_transforms=transforms.Resize((256, 256)),
)

defensed_models = [
    get_HGD_model(),
    Randomization(resnet_50),
    BitDepthReduction(resnet_50),
    JPEGCompression(resnet_50),
    randomized_smoothing_resnet50(),
    NeuralRepresentationPurifier(resnet_50),
    diffpure.eval().cuda(),
]
for model in defensed_models:
    model.eval().requires_grad_(False).cuda()
test_models += defensed_models

from torch.utils.data import DataLoader, TensorDataset

def eval_asr(path: str, target_model: nn.Module, batch_size=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> float:
    images = get_list_image(path, sort_key=lambda x: int(x[:-4]))  
    images = torch.stack(images).to(device)
    loader = get_NIPS17_loader(batch_size=len(images))
    batch = next(iter(loader))
    y_batch = batch[1].to(device)
    assert images.shape[0] == y_batch.shape[0]
    dataset = TensorDataset(images, y_batch)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct_counts = 0.0
    total_samples = 0
    with torch.no_grad():
        for img_batch, label_batch in data_loader:
            total_samples += img_batch.shape[0]
            pred = target_model(img_batch)
            pred = pred.argmax(dim=1)
            assert pred.shape == label_batch.shape
            correct_counts += (pred == label_batch).sum().item()
    return 1 - correct_counts / total_samples

def run_experiments_json(base_path: str, snum_list: List[int], models: List[nn.Module], json_file: str):
    results = {}
    model_names = ["adv_inception_v3", "ens_adv_inception_resnet_v2", "HGD_models", "Randomization", "BitDepthReduction", "JPEGCompression", "Randomized_Smoothing", "NeuralRepresentationPurifier", "DiffPure"]

    for model_name, model in zip(model_names, models):
        results[model_name] = {}
        for snum in snum_list:
            path = os.path.join(base_path, f"targeted/40_step_SSA_CWA_{snum}_snum_8_eps/adv")
            asrs = eval_asr(path, model)
            results[model_name][str(snum)] = asrs

    # Write results to the JSON file
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)

base_path = './results/classifiers'
snum_list = [1, 2, 4, 8, 16, 32, 64]

json_file = 'results_classifiers_defense.json'
run_experiments_json(base_path, snum_list, test_models, json_file)
