import numpy as np
import cv2

def find_object(new_img, old_img, thresh = 90, window_size = 15):
    # Blur
    blur_img = cv2.GaussianBlur(new_img, (window_size, window_size), 0)

    # Subtract new image from old one and take threshold
    dif = cv2.absdiff(blur_img, old_img)
    dif = np.sum(dif, axis=2, dtype=int)
    dif = np.where((dif > thresh), 255, 0).astype(np.uint8)

    # Filter threshold
    kernel = np.ones((window_size, window_size), np.uint8)
    img_thresh = cv2.erode(dif, kernel, iterations=2)
    img_thresh = cv2.dilate(img_thresh, kernel, iterations=6)

    # ROI
    img_thresh[:, :380] = 0
    img_thresh[:, 1220:] = 0
    img_thresh[660:, :] = 0

    # Find contours
    contours, hierarchy = cv2.findContours(image=img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    rects = []
    for c in contours:
        rect = cv2.boundingRect(c)
        area = rect[2] * rect[3]
        if area > 5000:
            rects.append((0, 0, (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])))

    # Find object
    return rects