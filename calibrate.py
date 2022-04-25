import numpy as np
import cv2
import glob, pickle
import matplotlib.pyplot as plt

def detect_chessboard(images_l, images_r):

    assert images_l
    assert images_r

    # Implement the number of vertical and horizontal corners
    nb_vertical = 9
    nb_horizontal = 6

    patternSize = (nb_vertical,nb_horizontal)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, patternSize[0]*patternSize[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints_l = [] # 3d point in real world space
    imgpoints_l = [] # 2d points in image plane.
    objpoints_r = [] # 3d point in real world space
    imgpoints_r = [] # 2d points in image plane.

    img_shape = None

    for i, fname in enumerate(images_l):
        img_l = cv2.imread(images_l[i])
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        img_shape = gray_l.shape
        
        retl, cornersl = cv2.findChessboardCorners(gray_l, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        img_r = cv2.imread(images_r[i])
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        retr, cornersr = cv2.findChessboardCorners(gray_r, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if retl is True:
            objpoints_l.append(objp)
            imgpoints_l.append(cornersl)
            
        if retr is True:
            objpoints_r.append(objp)
            imgpoints_r.append(cornersr)

    return objpoints_l, objpoints_r, imgpoints_l, imgpoints_r, img_shape

def calibrate_fishEye(objpoints, imgpoints, img_shape):
    # FishEye flags
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    # Calibrate Left
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    return K, D

def save_calibration(K_l, D_l, K_r, D_r):
    calib_dict = {'K_l':K_l, 'D_l':D_l, 'K_r':K_r, 'D_r':D_r}
    with open('data/calibration.pkl', 'wb') as handle:
        pickle.dump(calib_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_calibration(path):
    with open(path, 'rb') as handle:
        calib_dict = pickle.load(handle)

    return calib_dict['K_l'], calib_dict['D_l'], calib_dict['K_r'], calib_dict['D_r']


def undistort(img, K, D):
    DIM = img.shape[:2]
    DIM = DIM[::-1]
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


if __name__ == "__main__":
    # Image paths
    images_l = sorted(glob.glob('data/Stereo_calibration_images/left*.png'))
    images_r = sorted(glob.glob('data/Stereo_calibration_images/right*.png'))

    objpoints_l, objpoints_r, imgpoints_l, imgpoints_r, img_shape = detect_chessboard(images_l, images_r)

    K_l, D_l = calibrate_fishEye(objpoints_l, imgpoints_l, img_shape)
    K_r, D_r = calibrate_fishEye(objpoints_r, imgpoints_r, img_shape)

    save_calibration(K_l, D_l, K_r, D_r)


    # # Testing
    # K_l, D_l, K_r, D_r = load_calibration('data/calibration.pkl')
    # img_l = cv2.imread(images_l[45])
    # img_r = cv2.imread(images_r[45])

    # # undistort
    # dst_l = undistort(img_l, K_l, D_l)
    # dst_r = undistort(img_r, K_r, D_r)

    # # Plot
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
    # ax[0, 0].imshow(img_l[...,[2,1,0]])
    # ax[0, 0].set_title('Original image Left')
    # ax[0, 1].imshow(dst_l[...,[2,1,0]])
    # ax[0, 1].set_title('Undistorted image Left')

    # ax[1, 0].imshow(img_r[...,[2,1,0]])
    # ax[1, 0].set_title('Original image Right')
    # ax[1, 1].imshow(dst_r[...,[2,1,0]])
    # ax[1, 1].set_title('Undistorted image Right')

    # plt.show()