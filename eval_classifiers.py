import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (
    Swin_S_Weights, Swin_V2_B_Weights, Swin_V2_S_Weights,
    ConvNeXt_Base_Weights, ConvNeXt_Small_Weights,
    ResNeXt101_32X8D_Weights, ResNeXt50_32X4D_Weights
)
from typing import List, Callable
from data import get_NIPS17_loader
from utils import get_list_image, get_image
from models import *
import os
import json

# Measure transfer attack accuracy
def eval_asr(path: str,
                target_models: List[nn.Module],
                transform: Callable = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[float]:
    transfer_accs = [0.] * len(target_models)
    images = get_list_image(path, sort_key=lambda x: int(x[:-4]))
    images = [transform(image) for image in images]
    images = torch.stack(images).to(device)

    loader = get_NIPS17_loader(batch_size=len(images))
    batch = next(iter(loader))
    y_batch = batch[1].to(device)

    assert images.shape[0] == y_batch.shape[0]
    with torch.no_grad():
        for i, model in enumerate(target_models):
            model.eval()
            model.requires_grad_(False)
            pred = model(images)
            pred = pred.argmax(dim=1)
            assert pred.shape == y_batch.shape
            transfer_accs[i] = (pred == y_batch).float().mean().item()
    return [1 - acc for acc in transfer_accs]

# Measure the cross entropy loss with the target
def eval_loss(path: str,
            target_models: List[nn.Module],
            target_img_path: str,
            transform: Callable = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> List[float]:
    criterion = nn.CrossEntropyLoss(reduction='mean')
    transfer_losses = [0.] * len(target_models)
    accuracy_list = [0.] * len(target_models)
    images = get_list_image(path, sort_key=lambda x: int(x[:-4]))
    target = get_image(target_img_path)
    resizer = transforms.Resize((224, 224), antialias=None)
    target = resizer(target).to(device)
    if transform is not None:
        images = [transform(image) for image in images]
        target = transform(target)
    images = torch.stack(images).to(device)
    
    with torch.no_grad():
        for i, model in enumerate(target_models):
            model.eval()
            model.requires_grad_(False)
            pred = model(images)
            y_tar = model(target.unsqueeze(0)).argmax(dim=1)
            y_batch = torch.full((pred.size(0),), y_tar.item(), dtype=torch.long, device=device)

            loss = criterion(pred, y_batch)
            transfer_losses[i] = loss.item()

    return transfer_losses

# Run experiments and save as JSON
def run_experiments_json(base_path: str, snum_list: List[int], models: List[nn.Module], json_file: str, target_img_path: str):
    model_names = ["swin_s", "swin_v2_b", "swin_v2_s", "convnext_b", "convnext_s", "resnext101_32x8d", "resnext50_32x4d"]
    results = {name: {} for name in model_names}

    for snum in snum_list:
        path = os.path.join(base_path, f"targeted/40_step_SSA_CWA_{snum}_snum_8_eps/adv")
        print(f"Evaluating path: {path}")
        accs = eval_asr(path, models)
        losses = eval_loss(path, models, target_img_path)

        for name, acc in zip(model_names, accs):
            results[name][str(snum)] = [acc]
        for name, loss in zip(model_names, losses):
            results[name][str(snum)] += [loss]

    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    base_path = './results/classifiers'
    snum_list = [1, 2, 4, 8, 16, 32, 64]

    test_models = [
        swin_s(weights=Swin_S_Weights.DEFAULT).to('cuda'),
        swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT).to('cuda'),
        swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT).to('cuda'),
        convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT).to('cuda'),
        convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).to('cuda'),
        resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT).to('cuda'),
        resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT).to('cuda')
    ]

    json_file = 'results_classifiers.json'
    run_experiments_json(base_path, snum_list, test_models, json_file)