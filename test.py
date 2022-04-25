import torch
import numpy as np
import cv2

from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from utils.utils import collate_fn

from dataset import PerceptionDataset
from train import get_transform, get_model_object_detection


# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {1:'Books', 2:'Box', 3:'Mugs'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+1, 3))

def print_detection(img, detections):

    img = img.detach().cpu()
    img = np.array(ToPILImage()(img))
    print(img.shape)
    draw_img = img.astype(np.uint8).copy() 

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > 0.7:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(draw_img, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(draw_img, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return draw_img


def main():
    model = get_model_object_detection(4, False)
    model.load_state_dict(torch.load('data/models/model_weights.pth'))
    model.eval()

    dataset_test = PerceptionDataset('data/dataset', get_transform(train=False), json_name='test_torch.json')
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    for i, (img, target) in enumerate(data_loader_test):
        # img = img.to(DEVICE)
        detections = model(img)[0]
        result = print_detection(img[0], detections)

        cv2.imwrite('data/results/{}.jpg'.format(i), result)
        # cv2.waitKey(0)



if __name__ == "__main__":
    main()