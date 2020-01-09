"""post process trial and error (render and compare)"""

import numpy as np
import glob
import os
import cv2
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import json
import matplotlib.pylab as plt

# https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/car_instance/car_models.py
from car_models import car_name2id, car_id2name
from utils import get_intrinsics


def read_json(json_file):
    with open(json_file, 'r') as f_in:
        data = json.load(f_in)
    return data

def write_json(data, json_file):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w') as f_out:
        json.dump(data, f_out, sort_keys=True, indent=4)

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

def get_WHL(model_file):
    with open(model_file) as json_file:
        data = json.load(json_file)
    car_type = data['car_type']
    vertices = np.array(data['vertices'])
    triangles = np.array(data['faces']) - 1
    W, H, L = np.array(data['vertices']).max(axis=0) - np.array(data['vertices']).min(axis=0)
    model_name = os.path.basename(model_file).split('.json')[0]
    return W, H, L, car_type, model_name

def get_category_id_dict():
    category_id_dict = {}
    for x in car_id2name.values():
        category_id_dict[x.category] = x.categoryId
    return category_id_dict


# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
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


def draw_obj(image, vertices, triangles, color=(0,0,255)):
    for t in triangles:
        coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
#         cv2.fillConvexPoly(image, coord, color)
        # cv2.polylines(image, np.int32([coord]), 1, color, thickness=1)
    return image

# Load a 3D model of a car
def load_model_files(car_type):
    model_file = f'{basedir}/car_models_json/{car_type}.json'
    with open(model_file) as json_file:
        data = json.load(json_file)
    vertices = np.array(data['vertices'])
    vertices[:, 1] = -vertices[:, 1] ## y is pointing downward
    triangles = np.array(data['faces']) - 1
    return vertices, triangles

def get_mask(yaw, pitch, roll, x, y, z, model_id=None, model_class=None, color=(0, 0, 255), overlay=False, img=None):
    assert model_id or model_class
    if model_id:
        car_type = car_id2name[int(model_id)].name
    else:
        car_type = avg_size_dict[model_class]['model']
    vertices, triangles = load_model_files(car_type)
    yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
    # I think the pitch and yaw should be exchanged
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.ones((vertices.shape[0],vertices.shape[1]+1))
    P[:, :-1] = vertices
    P = P.T
    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    if not overlay:
        canvas = np.zeros_like(img)
    else:
        canvas = img
    canvas = draw_obj(canvas, img_cor_points, triangles, color=color)
    if overlay:
        alpha = .5
        img = np.array(img)
        cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, img)
    return canvas

def get_mask_bbox(yaw, pitch, roll, x, y, z, model_id=None, model_class=None):
    assert model_id or model_class
    if model_id:
        car_type = car_id2name[int(model_id)].name
    else:
        car_type = avg_size_dict[model_class]['model']
    vertices, triangles = load_model_files(car_type)
    yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
    # I think the pitch and yaw should be exchanged
    yaw, pitch, roll = -pitch, -yaw, -roll
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    P = np.ones((vertices.shape[0],vertices.shape[1]+1))
    P[:, :-1] = vertices
    P = P.T
    img_cor_points = np.dot(k, np.dot(Rt, P))
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    xmin, ymin = img_cor_points[:, :2].min(axis=0).astype(int)
    xmax, ymax = img_cor_points[:, :2].max(axis=0).astype(int)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, W-1)
    ymax = min(ymax, H-1)
    return (xmin, ymin), (xmax, ymax)

def get_bboxes_from_obj(obj):
    # get yaw, pitch, roll
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = np.array(obj['2D_bbox_xyxy']).astype(int)
    H, W, L = obj['3D_dimension']
    x, y, z = obj['3D_location']
    # in radian
    yaw = obj['global_yaw']
    pitch = obj['pitch']
    roll = -3.1
    if yaw > np.pi:
        yaw = yaw - 2 * np.pi
    yaw = -yaw
    yaw, pitch, roll = pitch, yaw, roll
    # now yaw, pitch, roll is GT, ready for plotting
    
    return (gt_xmin, gt_ymin, gt_xmax, gt_ymax), (yaw, pitch, roll, x, y, z)

def project_pred(img, pred, vis=False, optimize=False):
    """Overlay pred onto img"""
    for obj in (pred['objects'][:]):
        (gt_xmin, gt_ymin, gt_xmax, gt_ymax), (yaw, pitch, roll, x, y, z) = get_bboxes_from_obj(obj)
        
        # get initial projected mask and bbox
        model_class = '3x'
        (xmin, ymin), (xmax, ymax) = get_mask_bbox(yaw, pitch, roll, x, y, z, model_class=model_class)
        iou = get_iou(get_bb_dict(gt_xmin, gt_ymin, gt_xmax, gt_ymax), get_bb_dict(xmin, ymin, xmax, ymax))
    
        # optimize for x, y, z
        if optimize and z < 25:
            iou_max = 0
            best_xyz = None
            best_xyxy = None
            iou_init = iou
            x_init, y_init, z_init = x, y, z
            x_range = set([x_init] + list(np.linspace(x_init * 0.9, x_init * 1.1, 5)))  # 0.1 m interval
            y_range = set([y_init] + list(np.linspace(y_init * 0.9, y_init * 1.1, 5)))
            z_range = set([z_init] + list(np.linspace(z_init * 0.9, z_init * 1.1, 11)))
            for x in (x_range):
                for y in (y_range):
                    for z in (z_range):
                        (xmin, ymin), (xmax, ymax) = get_mask_bbox(
                            yaw, pitch, roll, x, y, z, model_class=model_class)
                        iou = get_iou(get_bb_dict(gt_xmin, gt_ymin, gt_xmax, gt_ymax), 
                                      get_bb_dict(xmin, ymin, xmax, ymax))
                        if iou > iou_max:
#                             print(f'{iou} > {iou_max}')
                            iou_max = iou
                            best_xyz = (x, y, z)
                            best_xyxy = xmin, ymin, xmax, ymax
            if best_xyz:
                x, y, z = best_xyz
                obj['3D_location'] = x, y, z  # write back to dict
                xmin, ymin, xmax, ymax = best_xyxy
                iou = iou_max
                print('{:.3f} --> {:.3f}: {:.3f}, {:.3f}, {:.3f}'.format(iou_init, iou, x - x_init, y - y_init, z - z_init))
            else:
                x, y, z = x_init, y_init, z_init
                
        # visualization
        if vis:
            canvas = get_mask(yaw, pitch, roll, x, y, z, 
                              model_id=None, model_class=model_class, overlay=True, img=img)
            canvas = cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
            canvas = cv2.rectangle(canvas, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (255, 0, 0), 10)
        else:
            canvas = None
    return iou, canvas, pred


if __name__ == '__main__':
    basedir = '/samsungSSD/datasets/pku-autonomous-driving/'
    model_files = glob.glob(os.path.join(basedir, 'car_models_json/*json'))

    category_id_dict = get_category_id_dict()
    W_list = []
    H_list = []
    L_list = []
    car_type_list = []
    model_name_list = []
    for model_file in model_files:
        W, H, L, car_type, model_name = get_WHL(model_file)
        W_list.append(W)
        H_list.append(H)
        L_list.append(L)
        car_type_list.append(car_type)
        model_name_list.append(model_name)

    # Build df_size
    size_list = np.array([W_list, H_list, L_list]).T
    category_id_list = [category_id_dict[x] for x in car_type_list]
    df_size = pd.DataFrame()
    df_size['W'] = W_list
    df_size['H'] = H_list
    df_size['L'] = L_list

    df_size['car_type'] = car_type_list
    df_size['category_id'] = category_id_list
    df_size['model_name'] = model_name_list

    df_size['volume'] = df_size['W'] * df_size['H'] * df_size['L']

    avg_size_dict = {}
    for c_type in df_size['car_type'].unique():
        avg_size_dict[c_type] = {}
        df_tmp = df_size[df_size['car_type'] == c_type]
        closest_to_med_idx = (df_tmp['volume'] - df_tmp.median()['volume']).abs().argsort().to_list()[0]
        median_model_name = df_tmp.iloc[closest_to_med_idx]['model_name']
        w, h, l, model_name = df_tmp.iloc[closest_to_med_idx][['W', 'H', 'L', 'model_name']]
        avg_size_dict[c_type]['W'] = w
        avg_size_dict[c_type]['H'] = h
        avg_size_dict[c_type]['L'] = l
        avg_size_dict[c_type]['model'] = model_name

    # Load an image
    img_name = 'ID_0ca978538'  # 'ID_0a0980d15'
    img = cv2.imread(f'{basedir}/train_images/{img_name}.jpg',cv2.COLOR_BGR2RGB)[:,:,::-1]
    img_raw = img.copy()
    H, W = img.shape[:2]

    train = pd.read_csv(f'{basedir}/train.csv')
    pred_string = train[train.ImageId == img_name].PredictionString.iloc[0]
    items = pred_string.split(' ')
    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
    xyz = np.array(list(zip(xs, ys, zs))).astype(float)


    # closest car
    closest_model_type_id = model_types[np.argmin(xyz[:, 2])]
    car_type_t = car_id2name[int(closest_model_type_id)].name

    # k is camera instrinsic matrix
    k = get_intrinsics()

    uuid_list = [os.path.basename(x)[:-4] for x in glob.glob('/home/xpilot/test_image_correct/*.jpg')]

    for uuid in tqdm(uuid_list[:]):
        image_file = '/home/xpilot/test_image_correct/{}.jpg'.format(uuid)
        json_file = '/home/xpilot/test_json_pred/{}.json'.format(uuid)
        json_file_refined = '/home/xpilot/test_json_pred_refined_coarse/{}.json'.format(uuid)
        pred = read_json(json_file)
        img = plt.imread(image_file)

        # optimize
        iou, canvas, pred = project_pred(img.copy(), pred, vis=False, optimize=True)
        write_json(pred, json_file_refined)


