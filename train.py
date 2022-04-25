from unicodedata import name
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
from utils.utils import collate_fn

import numpy as np
import cv2

from dataset import PerceptionDataset


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomZoomOut())
        transforms.append(T.ScaleJitter((800, 800)))

        
    return T.Compose(transforms)


def get_model_object_detection(num_classes, load_default=True):
    # load a model pre-trained on COCO
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=load_default)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=load_default)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4
    # use our dataset and defined transformations
    dataset = PerceptionDataset('data/dataset', get_transform(train=True))
    dataset_test = PerceptionDataset('data/dataset', get_transform(train=False))

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-10])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_object_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), 'data/models/model_weights.pth')
    print("That's it!")



if __name__ == "__main__":
    main()
