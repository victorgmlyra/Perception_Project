import torch
import numpy as np
import cv2, glob

from calibrate import load_calibration, stereo_undistort
from train import get_model_object_detection
from test import draw_detection
from depth import calculate_disparity, depth_heatmap
from detection import *

from kalman import *

# Set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = {0:'Random', 1:'Book', 2:'Box', 3:'Mug'}
font = cv2.FONT_HERSHEY_SIMPLEX

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

    # Frame with no objects
    l_first, r_first = None, None

    # Init Kalman
    x, P = init_kalman()

    # Video Writer
    video_out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (1280,720))

    min_size = (0,0)
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
        good_detections, img_t = find_object(orig_r, r_first)

        # Image Classification (DeepLearning)
        good_detections = classify(orig_r, good_detections, model, DEVICE)
        result = draw_detection(orig_r, good_detections)

        # TODO: Kalman Filter
        pos, min_size = get_objects_pos(good_detections, depth_map, CLASSES, min_size)

        if pos:
            z = np.array([[pos[0][1][0]],
                          [pos[0][1][1]],
                          [pos[0][1][2]]])
            x, P = update(x, P, z)
        
            # Show
            cv2.circle(result,(int(x[0][0]),int(x[3][0])),10,(0,255,0),2)
            cv2.putText(result, 'Current State', (int(x[0][0])+60,int(x[3][0])-60), font, 0.7, (0,255,0), 1, cv2.LINE_AA)
        else:
            min_size = (0,0)

        x, P, check, min_size = check_roi(x, P, min_size)
        if check:
            x, P = predict(x, P)
            cv2.circle(result,(int(x[0][0]),int(x[3][0])),10,(0,0,255),2)
            cv2.putText(result, 'Prediction', (int(x[0][0])-90,int(x[3][0])+90), font, 0.7, (0,0,255), 1, cv2.LINE_AA)

            cv2.putText(result, f'Predicted Position: {x[0][0]:.2f}, {x[3][0]:.2f}, {x[6][0]:.2f}', (0+90,0+90), font, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(result, f'Predicted Speed: {x[1][0]:.2f}, {x[4][0]:.2f}, {x[7][0]:.2f}', (0+90,0+120), font, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(result, f'Predicted Acceleration: {x[2][0]:.2f}, {x[5][0]:.2f}, {x[8][0]:.2f}', (0+90,0+150), font, 1, (0,0,0), 2, cv2.LINE_AA) 
        

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