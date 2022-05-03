import os, json
import torch
from PIL import Image
import utils.transforms as T
from torchvision.transforms import ToPILImage

import numpy as np

import cv2


class PerceptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, json_name = 'train_torch.json'):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        f = open(os.path.join(root, json_name))
        self.labels_dict = json.load(f)

        self.imgs = list(self.labels_dict.keys())

        f.close()

        self.encoding = {'Books':1, 'Box':2, 'Mugs':3}
        

    def __getitem__(self, idx):
        # load images and masks
        name = self.imgs[idx]
        img_path = os.path.join(self.root, name)

        img = Image.open(img_path).convert("RGB")

        # get bounding box coordinates for each image
        boxes = self.labels_dict[name]['rects']
        labels = [self.encoding[l] for l in self.labels_dict[name]['labels']]

        num_objs = len(labels)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


#  Transform function
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomZoomOut())
        transforms.append(T.ScaleJitter((800, 800)))

        
    return T.Compose(transforms)


if __name__ == "__main__":
    dataset = PerceptionDataset('data/dataset', get_transform(True))

    CLASSES = {1:'Books', 2:'Box', 3:'Mugs'}
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+1, 3))

    for tensor, boxes in dataset:
        img = tensor.detach().cpu()
        img = np.array(ToPILImage()(img))
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for idx in range(len(boxes['boxes'].numpy())):
            box = boxes['boxes'].numpy()[idx, :].astype("int")
            l = boxes['labels'].numpy()[idx]
            label = "{}".format(CLASSES[l])
            (startX, startY, endX, endY) = box
            cv2.rectangle(img, (startX, startY), (endX, endY),
                COLORS[l], 5)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[l], 2)
            

        cv2.imshow('a', img)
        key = cv2.waitKey(1000)
        if key == 27 or key == ord('q'):
            break