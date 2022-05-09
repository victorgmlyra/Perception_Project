from cv2 import edgePreservingFilter
import torch
import numpy as np
import cv2

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import PerceptionDataset, test_transforms
from train import get_model_object_detection

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASSES = {0:'Random', 1:'Book', 2:'Box', 3:'Mug'}
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+1, 3))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


return_transform = transforms.Compose(
    [
        UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.ToPILImage()
    ]
)

def draw_detection(img, good_detections, loader_img=False):
    draw_img = img.copy()
    # loop over the detections
    for detection in good_detections:
        idx, confidence, box = detection
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        # print("[INFO] {}".format(label))
        # draw the bounding box and label on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(draw_img, (startX, startY), (endX, endY),
            COLORS[idx], 3)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(draw_img, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idx], 2)
    return draw_img


def main():
    # Load trained model
    model, _ = get_model_object_detection(4, False)
    model.load_state_dict(torch.load('data/models/model_weights.pth'))
    model.eval()

    # Test dataset loader
    dataset_test = PerceptionDataset('data/dataset', test_transforms)
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4)

    # Iterate over test dataset
    samples = len(dataset_test) + 7
    rows = samples // 8
        
    figure, ax = plt.subplots(nrows=rows, ncols=8, figsize=(24, 16))
    y_test, y_pred = [], []
    for i, (img, label) in enumerate(data_loader_test):
        output = model(img)
        index = output.data.cpu().numpy().argmax()
        pil_img = return_transform(img.squeeze(0))
        cv_img = np.array(pil_img)
        
        ax.ravel()[i].imshow(cv_img)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(dataset_test.idx_to_class[index])

        y_test.append(label.numpy()[0])
        y_pred.append(index)

    plt.tight_layout(pad=1)
    figure.delaxes(ax[3][7])

    # Confussion Matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(2)
    ax2 = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax2.set_title('Confusion Matrix \n')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax2.xaxis.set_ticklabels(dataset_test.idx_to_class.values())
    ax2.yaxis.set_ticklabels(dataset_test.idx_to_class.values())

    # Show plots
    plt.show()  

if __name__ == "__main__":
    main()