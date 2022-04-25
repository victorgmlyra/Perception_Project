import numpy as np
import cv2
import glob, pickle


def detect_chessboard(images_l, images_r):

    assert images_l
    assert images_r

    # Implement the number of vertical and horizontal corners
    nb_vertical = 9
    nb_horizontal = 6

    patternSize = (nb_vertical,nb_horizontal)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((patternSize[0]*patternSize[1], 1, 3), np.float32)
    objp[:,0,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

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
        if retl is True and retr is True:
            objpoints_l.append(objp)
            imgpoints_l.append(cornersl)
            
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
    rms, camera_matrix, dist_coeffs, _, _ = \
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

    # Rectify
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img_shape[::-1], cv2.CV_16SC2)
    
    return map1, map2, camera_matrix, dist_coeffs

def calibrate_stereo_fisheye(objpoints, imgpoints_l, camera_matrix_l, dist_coeffs_l, imgpoints_r, camera_matrix_r, dist_coeffs_r, img_shape):
    TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    OPTIMIZE_ALPHA = 0.25

    objpoints = np.array(objpoints)
    imgpoints_l = np.asarray(imgpoints_l)
    imgpoints_r = np.asarray(imgpoints_r)

    (RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            camera_matrix_l, dist_coeffs_l,
            camera_matrix_r, dist_coeffs_r,
            img_shape[::-1], None, None,
            cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

    # Rectifying
    R1 = np.zeros([3,3])
    R2 = np.zeros([3,3])
    P1 = np.zeros([3,4])
    P2 = np.zeros([3,4])
    Q = np.zeros([4,4])

    (leftRectification, rightRectification, leftProjection, rightProjection,
            dispartityToDepthMap) = cv2.fisheye.stereoRectify(
                    camera_matrix_l, dist_coeffs_l,
                    camera_matrix_r, dist_coeffs_r,
                    img_shape[::-1], rotationMatrix, translationVector,
                    0, R2, P1, P2, Q,
                    cv2.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)

    # Saving calibration results for the future use
    print("Saving calibration...")
    leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix_l, dist_coeffs_l, leftRectification,
            leftProjection, img_shape[::-1], cv2.CV_16SC2)
    rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix_r, dist_coeffs_r, rightRectification,
            rightProjection, img_shape[::-1], cv2.CV_16SC2)

    return leftMapX, leftMapY, rightMapX, rightMapY, dispartityToDepthMap



def save_calibration(leftMapX, leftMapY, rightMapX, rightMapY, dispartityToDepthMap):
    calib_dict = {'leftMapX': leftMapX, 'leftMapY': leftMapY, 
                  'rightMapX': rightMapX, 'rightMapY': rightMapY, 
                  'dispartityToDepthMap': dispartityToDepthMap}
    with open('data/calibration.pkl', 'wb') as handle:
        pickle.dump(calib_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_calibration(path):
    with open(path, 'rb') as handle:
        calib_dict = pickle.load(handle)

    return calib_dict


def stereo_undistort(img_l, img_r, calib_dict):
    leftMapX, leftMapY = calib_dict['leftMapX'], calib_dict['leftMapY']
    rightMapX, rightMapY = calib_dict['rightMapX'], calib_dict['rightMapY']
    dst_l = cv2.remap(img_l, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst_r = cv2.remap(img_r, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return dst_l, dst_r


if __name__ == "__main__":
    # Image paths
    images_l = sorted(glob.glob('data/Stereo_calibration_images/left*.png'))
    images_r = sorted(glob.glob('data/Stereo_calibration_images/right*.png'))

    objpoints_l, objpoints_r, imgpoints_l, imgpoints_r, img_shape = detect_chessboard(images_l, images_r)

    map1_l, map2_l, camera_matrix_l, dist_coeffs_l = calibrate_fishEye(objpoints_l, imgpoints_l, img_shape)
    map1_r, map2_r, camera_matrix_r, dist_coeffs_r = calibrate_fishEye(objpoints_r, imgpoints_r, img_shape)

    leftMapX, leftMapY, rightMapX, rightMapY, dispartityToDepthMap = calibrate_stereo_fisheye(objpoints_l, 
                                                                    imgpoints_l, camera_matrix_l, dist_coeffs_l, 
                                                                    imgpoints_r, camera_matrix_r, dist_coeffs_r, img_shape)

    save_calibration(leftMapX, leftMapY, rightMapX, rightMapY, dispartityToDepthMap)


    # # Testing
    # calibration = load_calibration('data/calibration.pkl')
    # img_l = cv2.imread(images_l[45])
    # img_r = cv2.imread(images_r[45])

    # # undistort
    # dst_l, dst_r = stereo_undistort(img_l, img_r, calibration)

    # # Plot
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
    # ax[0, 0].imshow(img_l[...,[2,1,0]])
    # ax[0, 0].set_title('Original image Left')
    # ax[1, 0].imshow(dst_l[...,[2,1,0]])
    # ax[1, 0].set_title('Undistorted image Left')

    # ax[0, 1].imshow(img_r[...,[2,1,0]])
    # ax[0, 1].set_title('Original image Right')
    # ax[1, 1].imshow(dst_r[...,[2,1,0]])
    # ax[1, 1].set_title('Undistorted image Right')

    # plt.show()