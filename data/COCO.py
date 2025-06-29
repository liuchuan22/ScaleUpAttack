from torch.utils.data import Dataset, DataLoader
import csv
import os
from PIL import Image
from torchvision import transforms

class COCO(Dataset):
    # please enter COCO data path
    def __init__(self, images_path='./COCO_100/val2017',
                 transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                 ]),
                 ):
        self.images = os.listdir(images_path)
        self.images.sort()
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        leng = len(self.images)
        assert leng == 100, "more than 100 images in the dataset!"
        return leng

    def __getitem__(self, item):
        name = self.images[item]
        x = Image.open(os.path.join(self.images_path, name))
        return self.transform(x), ""


def get_COCO_loader(batch_size=64,
                      num_workers=8,
                      pin_memory=True,
                      shuffle=False,
                      ):
    set = COCO()
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    return loader
