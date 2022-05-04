import copy
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import cv2, glob

class PerceptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, train = False):
        self.root = root
        self.transform = transforms

        self.class_to_idx = {'Random':0, 'Book':1, 'Box':2, 'Mug':3}
        self.idx_to_class = {value:key for key,value in self.class_to_idx.items()}

        # load all image files, sorting them to
        # ensure that they are aligned
        if train:
            path = root + '/train'
        else:
            path = root + '/test'

        self.img_paths = glob.glob(path + '/*/*')
        

    def __getitem__(self, idx):
        # load images and masks
        name = self.img_paths[idx]

        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = name.split('/')[-2]
        label = self.class_to_idx[label]

        if self.transform is not None:
            img = self.transform(image=image)["image"]

        return img, label

    def __len__(self):
        return len(self.img_paths)


train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=300),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=224, width=224),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=257),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img = False):
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(dataset))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(dataset.idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = PerceptionDataset('data/dataset', train_transforms, train=True)

    visualize_augmentations(dataset,np.random.randint(1,len(dataset)), random_img = True)