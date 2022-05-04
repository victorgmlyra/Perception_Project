import numpy as np
import cv2

from dataset import test_transforms

def find_object(new_img, old_img, thresh = 70, window_size = 11):
    # Blur
    blur_img = cv2.GaussianBlur(new_img, (window_size, window_size), 0)

    # Subtract new image from old one and take threshold
    dif = cv2.absdiff(blur_img, old_img)
    dif = np.sum(dif, axis=2, dtype=int)
    dif = np.where((dif > thresh), 255, 0).astype(np.uint8)

    # ROI
    dif[:, :370] = 0
    dif[:, 1230:] = 0
    dif[660:, :] = 0
    dif[420:500, 1015:1100] = 0
    dif[390:450, 1140:1200] = 0

    # Filter threshold
    kernel = np.ones((window_size, window_size), np.uint8)
    img_thresh = cv2.erode(dif, kernel, iterations=1)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=4)
    # img_thresh = cv2.erode(dif, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(image=img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    rects = []
    # Find object
    for c in contours:
        rect = cv2.boundingRect(c)
        area = rect[2] * rect[3]
        if area > 5000:
            rects.append((0, 0, (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])))

    return rects, dif


def softmax(x):
    # Turn results into percentages
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def classify(img, detections, model, device):
    classification = []
    for d in detections:
        # Crop image to object
        x0, y0, x1, y1 = d[2]
        crop_img = img[y0:y1, x0:x1]

        # Tranform image to tensor
        tensor = test_transforms(image=crop_img)['image']
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        # Run image classifier
        output = model(tensor)
        values = softmax(output.data.cpu().numpy())
        index = values.argmax()

        classification.append((index, values[0, index], d[2]))

    return classification
