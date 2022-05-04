import torch
import numpy as np
import cv2, glob

from calibrate import load_calibration, stereo_undistort
from train import get_model_object_detection
from test import draw_detection
from depth import calculate_disparity, depth_heatmap
from detection import *

# Set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {0:'Random', 1:'Book', 2:'Box', 3:'Mug'}

def get_objects_pos(detections, depth_map):
    objs = []
    for d in detections:
        idx, confidence, box = d
        c = CLASSES[idx]
        (startX, startY, endX, endY) = box
        mid_point = [int((startX+endX)/2), int((startY+endY)/2)]
        # Z axis
        bb_img = depth_map[startY:endY, startX:endX]
        min_z = np.min(bb_img)
        mid_point.append(min_z)
        objs.append((c, mid_point))
    return objs


def main():
    # Loading Object Detection Model
    model, _ = get_model_object_detection(len(CLASSES), False)
    model.load_state_dict(torch.load('data/models/model_weights.pth'))
    model.eval()
    model.to(DEVICE)

    # Loading Image Paths
    img_l_paths = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/left/*'))
    img_r_paths = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/right/*'))

    # Loading Undistortion parameters
    calibration = load_calibration('data/calibration.pkl')

    l_first, r_first = None, None
    video_out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1280,720))
    for i, (l_path, r_path) in enumerate(zip(img_l_paths, img_r_paths)):

        # Loading Images
        orig_l = cv2.imread(l_path)
        orig_r = cv2.imread(r_path)

        # Undistort Images
        orig_l, orig_r = stereo_undistort(orig_l, orig_r, calibration)
        if i < 40:
            l_first = cv2.GaussianBlur(orig_l, (7, 7), 0)
            r_first = cv2.GaussianBlur(orig_r, (7, 7), 0)

        # Depth Estimation
        disparity = calculate_disparity(orig_l, orig_r)
        depth_map = cv2.reprojectImageTo3D(disparity, calibration['dispartityToDepthMap'])[:, :, 2] 
        depth_map_show = depth_heatmap(depth_map, 5)

        # Object Detection (Image Processing)
        good_detections = find_object(orig_l, l_first)

        # Image Classification (DeepLearning)
        good_detections = classify(orig_l, good_detections, model, DEVICE)
        result = draw_detection(orig_l, good_detections)

        # TODO: Kalman Filter
        pos = get_objects_pos(good_detections, depth_map)
        print(pos)

        # Show
        cv2.imshow("Detection Result", result)
        video_out.write(result)
        cv2.imshow("Depth Map", depth_map_show)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        
    video_out.release()



if __name__ == "__main__":
    main()