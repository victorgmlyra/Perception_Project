import torch
import numpy as np
import cv2, glob

from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from utils.utils import collate_fn

from dataset import PerceptionDataset
from train import get_transform, get_model_object_detection


# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {1:'Books', 2:'Box', 3:'Mugs'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+1, 3))


def main():
    model = get_model_object_detection(4, False)
    model.load_state_dict(torch.load('data/models/model_weights.pth'))
    model.eval()
    model.to(DEVICE)
    
    img_paths = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/right/*'))
    #img_first = cv2.imread(img_paths[0])

    for img_path in img_paths:
    
        image = cv2.imread(img_path)
        # image -= img_first
        orig = image.copy()
        

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)

        image = image.to(DEVICE)
        detections = model(image)[0]

        # loop over the detections
        for i in range(0, len(detections["boxes"])):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections["scores"][i]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                # disconfidenceplay the prediction to our terminal
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                    COLORS[idx], 7)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow("Output", orig)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()