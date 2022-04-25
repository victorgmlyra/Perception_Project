import torch
import numpy as np
import cv2, os

from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from utils.utils import collate_fn

from dataset import PerceptionDataset
from train import get_transform, get_model_object_detection
from json_transform import json_transform


# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {1:'Books', 2:'Box', 3:'Mugs'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+1, 3))

def get_good_detections(detections, min_confidence=0.6):
    good_detections = []
    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > min_confidence:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            good_detections.append((idx, confidence, box.astype("int")))

    return good_detections

def draw_detection(img, good_detections, loader_img=False):
    if loader_img:
        img = img.detach().cpu()
        img = np.array(ToPILImage()(img))
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    draw_img = img.copy() 
    # loop over the detections
    for detection in good_detections:
        idx, confidence, box = detection
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        # draw the bounding box and label on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(draw_img, (startX, startY), (endX, endY),
            COLORS[idx], 5)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(draw_img, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
    return draw_img


def main():
    # Transform JSON
    json_transform('test')

    # Load trained model
    model = get_model_object_detection(4, False)
    model.load_state_dict(torch.load('data/models/model_weights.pth'))
    model.eval()

    # Test dataset loader
    dataset_test = PerceptionDataset('data/dataset', get_transform(train=False), json_name='test_torch.json')
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # Iterate over test dataset
    for i, (img, target) in enumerate(data_loader_test):
        detections = model(img)[0]
        good_detections = get_good_detections(detections, 0.6)
        result = draw_detection(img[0], good_detections, True)

        cv2.imwrite('data/results/{}.jpg'.format(i), result)


if __name__ == "__main__":
    if not os.path.isdir('data/results'):
        os.mkdir('data/results')
    main()