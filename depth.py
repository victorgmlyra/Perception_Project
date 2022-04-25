import numpy as np 
import cv2
import matplotlib.pyplot as plt

from calibrate import *

numDisparities = 160
blockSize = 17
preFilterType = 0
preFilterSize = 19
preFilterCap = 26
textureThreshold = 11
uniquenessRatio = 16
speckleRange = 16
speckleWindowSize = 26
disp12MaxDiff = 2
minDisparity = 2


def calculate_disparity(img_l, img_r):
    Left_nice = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    Right_nice = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()

    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Calculating disparity
    disparity = stereo.compute(Left_nice,Right_nice)
    disparity = disparity.astype(np.float32)
    # disparity = (disparity/16.0 - minDisparity)/numDisparities

    return disparity


def depth_heatmap(depth_map, max_clip=10):
    clipped = np.clip(depth_map, 0, max_clip).astype(np.uint8)
    depth_map_show = cv2.normalize(clipped, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    colormap = plt.get_cmap('inferno')
    depth_map_show = (colormap(depth_map_show) * 2**16).astype(np.uint16)[:,:,:3]
    depth_map_show = cv2.cvtColor(depth_map_show, cv2.COLOR_RGB2BGR)
    return depth_map_show



if __name__ == '__main__':
    # Loading Undistortion parameters
    calibration = load_calibration('data/calibration.pkl')

    Left_Stereo_Map_x = calibration['leftMapX']
    Left_Stereo_Map_y = calibration['leftMapY']
    Right_Stereo_Map_x = calibration['rightMapX']
    Right_Stereo_Map_y = calibration['rightMapY']

    # Loading Image Paths
    img_l_paths = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/left/*'))
    img_r_paths = sorted(glob.glob('data/Stereo_conveyor_without_occlusions/right/*'))


    def nothing(x):
        pass

    cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('disp',600,600)

    cv2.createTrackbar('numDisparities','disp',int(numDisparities/16),17,nothing)
    cv2.createTrackbar('blockSize','disp',int((blockSize-5)/2),50,nothing)
    cv2.createTrackbar('preFilterType','disp',preFilterType,1,nothing)
    cv2.createTrackbar('preFilterSize','disp',int((preFilterSize-5)/2),25,nothing)
    cv2.createTrackbar('preFilterCap','disp',preFilterCap,62,nothing)
    cv2.createTrackbar('textureThreshold','disp',textureThreshold,100,nothing)
    cv2.createTrackbar('uniquenessRatio','disp',uniquenessRatio,100,nothing)
    cv2.createTrackbar('speckleRange','disp',speckleRange,100,nothing)
    cv2.createTrackbar('speckleWindowSize','disp',speckleWindowSize,25,nothing)
    cv2.createTrackbar('disp12MaxDiff','disp',disp12MaxDiff,25,nothing)
    cv2.createTrackbar('minDisparity','disp',minDisparity,25,nothing)

    # Creating an object of StereoBM algorithm
    stereo = cv2.StereoBM_create()

    for l_path, r_path in zip(img_l_paths, img_r_paths):
        # Loading Images
        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)
        
        # Proceed only if the frames have been captured
        imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

        Left_nice, Right_nice = stereo_undistort(imgL_gray, imgR_gray, calibration)

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
        blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
        speckleRange = cv2.getTrackbarPos('speckleRange','disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
        minDisparity = cv2.getTrackbarPos('minDisparity','disp')
        
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice,Right_nice)
        # NOTE: Code returns a 16bit signed single channel image,
        # CV_16S containing a disparity map scaled by 16. Hence it 
        # is essential to convert it to CV_32F and scale it down 16 times.

        # Converting to float32 
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them 
        disparity = (disparity/16.0 - minDisparity)/numDisparities

        # Displaying the disparity map
        cv2.imshow("disp",disparity)

        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break
        
    # Updating the parameters based on the trackbar positions
    print('numDisparities =', numDisparities)
    print('blockSize =', blockSize) 
    print('preFilterType =', preFilterType) 
    print('preFilterSize =', preFilterSize)
    print('preFilterCap =', preFilterCap)
    print('textureThreshold =', textureThreshold)
    print('uniquenessRatio =', uniquenessRatio)
    print('speckleRange =', speckleRange) 
    print('speckleWindowSize =', speckleWindowSize)
    print('disp12MaxDiff =', disp12MaxDiff)
    print('minDisparity =', minDisparity)
