import os, glob, cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path as osp
from car_models import car_name2id, car_id2name
import numpy as np
from math import sin, cos
from PIL import ImageDraw, Image


def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_bb_dict(xmin, ymin, xmax, ymax):
    bb_dict = {
        'x1': xmin,
        'x2': xmax,
        'y1': ymin,
        'y2': ymax
    }
    return bb_dict

def read_json(json_file):
    with open(json_file, 'r') as f_in:
        data = json.load(f_in)
    return data

def write_json(data, json_file):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f_out:
        json.dump(data, f_out, sort_keys=True, indent=4)

def load_3dlabel(car_model, model_dir=None):
    model_dir = '../data/car_models_json' if model_dir is None else model_dir
    try:
        car_name = car_id2name[int(car_model)][0]
    except:
        car_name = car_model

    json_file = open('{}/{}.json'.format(model_dir, car_name), 'rb')
    data = json.load(json_file)
    return data, car_name

class NumpyEncoder(json.JSONEncoder):
    """Helper class to help serialize numpy ndarray"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# convert euler angle to rotation matrix
def euler_to_rot(yaw, pitch, roll):
    Y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, np.cos(pitch), -np.sin(pitch)],
                  [0, np.sin(pitch), np.cos(pitch)]])
    R = np.array([[np.cos(roll), -np.sin(roll), 0],
                  [np.sin(roll), np.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 5)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 5)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 5)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 5)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 5)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 5)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 5)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 5)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 5)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 5)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 5)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 5)
    return image

def draw_points(image, points):
    for idx, (p_x, p_y, p_z) in enumerate(points):
        if idx == 0:
            color = (0, 0, 255)
            size = 1
        else:
            color = (0, 0, 255)
            size = 5
        if idx == 1:
            color = (0, 255, 0)
            size = 5
        if idx == 7:
            color = (255, 0, 0)
            size = 5
        cv2.circle(image, (p_x, p_y), size, color, -1)
    return image

# image coordinate to world coordinate
def img_cor_2_world_cor(img_cor_points, k):
    x_img, y_img, z_img = img_cor_points[0]
    xc, yc, zc = x_img*z_img, y_img*z_img, z_img
    p_cam = np.array([xc, yc, zc])
    xw, yw, zw = np.dot(np.linalg.inv(k), p_cam)
    print(xw, yw, zw)
    print(x, y, z)

def fill_hole(image):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    contour,hier = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(gray,[cnt],0,255,-1)
    return gray

def draw_obj(image, vertices, triangles, color=(0,0,255)):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
#         cv2.fillConvexPoly(image, coord, color)
        cv2.polylines(image, np.int32([coord]), 1, color, thickness=1)
        # image = fill_hole(image)
    return image

def get_intrinsics():
    k = np.array([[2304.5479, 0, 1686.2379],
                  [0, 2305.8757, 1354.9849],
                  [0, 0, 1]], dtype=np.float32)
    return k

def get_avg_size():
    avg_dict = {'2x': {'W': 1.81794264,
                'H': 1.47786305,
                'L': 4.49547776,
                'model': 'bieke-yinglang-XT'},
         'SUV': {'W': 2.10604523,
                 'H': 1.67994469,
                 'L': 4.73350861,
                 'model': 'biyadi-tang'},
         '3x': {'W': 1.9739563700000002,
                'H': 1.4896684399999998,
                'L': 4.83009344,
                'model': 'dazhongmaiteng'}}
    return avg_dict

if __name__ == '__main__':

    k = get_intrinsics()
    image_dir = '/Users/pliu/Downloads/pku_kaggle_data/train_images'
    pose_csv = '/Users/pliu/Downloads/pku_kaggle_data/train_images/train.csv'
    
    img_files = glob.glob(osp.join(image_dir, '*jpg'))
    print(img_files)
    img_files.sort()
    for img_file in img_files[:1]:
        print(img_file)

        pid = osp.basename(img_file).split('.jpg')[0]
        json_path = osp.join(pose_dir, pid + '.json')
        keypoint_path = osp.join(keypoint_dir, pid)
        if osp.exists(json_path):
            with open(json_path, 'r') as fin:
                pose = json.load(fin)
        if osp.exists(keypoint_dir):
            txt_files = glob.glob(osp.join(keypoint_path, '*txt'))

        for txt_file in txt_files:
            file = open(txt_file, 'r')
            for line in file.readlines():
                print(line.split())

        im = cv2.imread(img_file)
        plt.imshow(im)
        plt.show()
        overlays = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        for obj in pose:
            car_id = obj['car_id']
            roll, pitch, yaw, x, y, z = obj['pose']
            yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]

            yaw, pitch, roll = -pitch, -yaw, -roll

            # roll = roll + np.pi
            # pitch = -pitch
            # yaw = -yaw
            data, car_name = load_3dlabel(car_id)


            vertices = np.array(data['vertices'])
            vertices[:, 1] = -vertices[:, 1]
            triangles = np.array(data['faces']) - 1
            w, h, l = vertices.max(axis=0) - vertices.min(axis=0)

            Rt = np.eye(4)
            t = np.array([x, y, z])
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            P[:, :-1] = vertices
            P = P.T
            img_cor_points = np.dot(k, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]

            overlay = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            overlay = draw_obj(overlay, img_cor_points, triangles)
            # plt.imshow(overlay)
            # plt.show()
            overlays = overlay + overlays
        plt.imshow(overlays)
        plt.show()


            # xmin = img_cor_points[:, 0].min()
            # xmax = img_cor_points[:, 0].max()
            # ymin = img_cor_points[:, 1].min()
            # ymax = img_cor_points[:, 1].max()
            # xc = (xmin + xmax) // 2
            # yc = (ymin + ymax) // 2
            # w = xmax - xmin
            # h = ymax - ymin
            # obj['2D_bbox_xyxy'] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            # obj['2D_bbox_xywh'] = [xc, yc, w, h]
            #
        # image = Image.fromarray(img)
        # # alpha = .5
        # # img = np.array(img)
        # # cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # plt.figure(figsize=(20, 20))
        # plt.imshow(img)

