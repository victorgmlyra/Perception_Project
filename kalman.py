from tracemalloc import start
import numpy as np
import cv2

### Kalman filter variables ###
# The external motion (6x1).
# The external motion (9x1).
u = np.array([[0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0]])

# The transition matrix (9x9). 
F = np.array([[1, 1, 0.5, 0, 0,  0,  0, 0,  0 ],
              [0, 1,  1 , 0, 0,  0,  0, 0,  0 ],
              [0, 0,  1 , 0, 0,  0,  0, 0,  0 ],
              [0, 0,  0 , 1, 1, 0.5, 0, 0,  0 ],
              [0, 0,  0 , 0, 1,  1,  0, 0,  0 ],
              [0, 0,  0 , 0, 0,  1,  0, 0,  0 ],
              [0, 0,  0 , 0, 0,  0,  1, 1, 0.5],
              [0, 0,  0 , 0, 0,  0,  0, 1,  1 ],
              [0, 0,  0 , 0, 0,  0,  0, 0,  1 ]])

# The observation matrix (3x9).
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0]])

# The measurement uncertainty (3x3).
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])



def update(x, P, Z):
    ### Insert update function
    I = np.eye(9, 9)
    
    y = Z - np.dot(H, x)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))

    updated_x = x + np.dot(K, y)
    updated_P = np.dot((I - np.dot(K, H)), P)
    #updated_P = np.dot(np.dot((I - np.dot(K, H)), P), np.transpose((I - np.dot(K, H)))) + np.dot(np.dot(K, R), np.transpose(K))

    return updated_x, updated_P


def predict(x, P):
    ### insert predict function
    predicted_x = np.dot(F, x) + u
    predicted_P = np.dot(np.dot(F, P), np.transpose(F))

    return predicted_x, predicted_P


# Initialize Kalman filter
def init_kalman():
    # The initial state (9x1).
    x = np.array([[0], # Position along the x-axis
                [0], # Velocity along the x-axis
                [0], # Acceleration along the x-axis
                [0], # Position along the y-axis
                [0], # Velocity along the y-axis
                [0], # Acceleration along the y-axis
                [0], # Position along the z-axis
                [0], # Velocity along the z-axis
                [0]])# Acceleration along the z-axis

    # The initial uncertainty (9x9).
    P = np.array([[1000, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1000, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1000, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1000, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1000, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1000, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1000, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1000, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1000]])
    return x, P


def check_roi(x, P, min_size):
    check = True

    if x[0][0] < 300 or x[0][0] > 1050 or x[3][0] > 600 or x[3][0] < 250:
        check = False
        x, P = init_kalman()
        min_size = (0,0)

    return x, P, check, min_size


def get_objects_pos(detections, depth_map, CLASSES, min_size):
    objs = []
    min_w, min_h = 0, 0
    for d in detections:
        idx, confidence, box = d
        c = CLASSES[idx]
        (startX, startY, endX, endY) = box

        if endX - startX < min_size[0]:
            startX = endX - min_size[0]
            min_w = min_size[0]
        else:
            min_w = endX - startX
        # if endY - startY < min_size[1]:
        #     endY = startY + min_size[1]
        #     min_h = min_size[1]
        # else:
        #     min_h = endY - startY

        mid_point = [int((startX+endX)/2), int((startY+endY)/2)]
        # Z axis
        bb_img = depth_map[startY:endY, startX+100:endX+100]
        min_z = np.min(bb_img)
        mid_point.append(min_z)
        objs.append((c, mid_point))

    return objs, (min_w, min_h)




