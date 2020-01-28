import os
import cv2
import glob
import pandas as pd
import json
import numpy as np
from matplotlib import pylab as plt
import argparse
from tqdm import tqdm
from utils import get_avg_size, draw_obj, euler_to_rot, get_intrinsics
from car_models import car_id2name
from utils import get_avg_size, get_intrinsics


# Load a 3D model of a car
def load_model_files(car_type):
    model_file = f'{basedir}/car_models_json/{car_type}.json'
    with open(model_file) as json_file:
        data = json.load(json_file)
    vertices = np.array(data['vertices'])
    # vertices[:, 1] = -vertices[:, 1] ## y is pointing downward
    triangles = np.array(data['faces']) - 1
    return vertices, triangles


def get_mask(yaw, pitch, roll, x, y, z, model_id=None, model_class=None,
             color=(0, 0, 255), overlay=False, img=None):
    assert model_id or model_class
    if model_id:
        car_type = car_id2name[int(model_id)].name
    else:
        car_type = avg_size_dict[model_class]['model']
    vertices, triangles = load_model_files(car_type)
    yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
    Rt = np.eye(4)
    t = np.array([x, y, z])
    Rt[:3, 3] = t
    Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll)
    Rt = Rt[:3, :]
    P = np.ones((vertices.shape[0],vertices.shape[1]+1))
    P[:, :-1] = vertices
    # P[:, 1] *= -1  # y in models are flipped
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


# properly defined
# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    P = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    R = np.array([[1, 0, 0],
                  [0, np.cos(pitch), -np.sin(pitch)],
                  [0, np.sin(pitch), np.cos(pitch)]])
    Y = np.array([[np.cos(roll), -np.sin(roll), 0],
                  [np.sin(roll), np.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


if __name__ == '__main__':
    avg_size_dict = get_avg_size()

    intrinsics_params = {
      'fpx': 592.85791,
      'fpy': 592.85791,
      'cx': 639.6357421875,
      'cy': 478.307465,
    }
    k = get_intrinsics()
    k[0, 0] = intrinsics_params['fpx']
    k[1, 1] = intrinsics_params['fpy']
    k[0, 2] = intrinsics_params['cx']
    k[1, 2] = intrinsics_params['cy']

    basedir = './data/'

    # canvas size
    width, height = int(k[0, 2] * 2), int(k[1, 2] * 2)

    yaw, pitch, roll = 1.25, 0, 0
    x, y, z = (-0.267754733, 0.624151607, 3.15893658)
    canvas = get_mask(yaw, pitch, roll, x, y, z, model_class='3x',
                      img=np.zeros((height, width, 3), np.uint8), overlay=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(canvas)
    plt.show()