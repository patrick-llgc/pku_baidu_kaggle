"""This file converts apolloScape dataset to kaggle format"""
import numpy as np
import math
import glob
from tqdm import tqdm
import os
import json
from scipy.spatial.transform import Rotation as R


def euler_angles_to_rotation_matrix(angle, degrees=False, is_dir=False):
    """Convert euler angels to quaternions.

    This is from apolloScape api
    from https://github.com/stevenwudi/6DVNET/blob/master/tools/ApolloScape_car_instance/utils/utils.py#L69

    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]
    if degrees:
        roll, pitch, yaw = [np.deg2rad(x) for x in [roll, pitch, yaw]]

    # roll: around x-axis
    # pitch: around y-axis
    # yaw: around z-axis
    # rotmat: roll first, then pitch, then yaw, so 'xyz' in euler seq
    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R

# convert euler angle to rotation matrix
def euler_to_rot(yaw, pitch, roll, degrees=False):
    if degrees:
        roll, pitch, yaw = [-np.deg2rad(x) for x in [roll, pitch, yaw]]

    Y = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                  [0, 1, 0],
                  [-np.sin(yaw), 0, np.cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, np.cos(pitch), -np.sin(pitch)],
                  [0, np.sin(pitch), np.cos(pitch)]])
    R = np.array([[np.cos(roll), -np.sin(roll), 0],
                  [np.sin(roll), np.cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R)).T


def convert_apollo_angle_to_kaggle(angle, degrees=False):
    """Convert euler angles from apolloScape format to kaggle format
    Order: roll, pitch, yaw = angle"""
    r = R.from_dcm(euler_angles_to_rotation_matrix(angle, degrees=degrees))
    angle = r.as_euler('yxz', degrees=degrees)
    return angle

# sanity check
def test_convert_apollo_to_kaggle():
    angle = [-8, 21.45, 30]
    kaggle_angle = convert_apollo_angle_to_kaggle(angle, degrees=True)
    r1 = R.from_dcm(euler_angles_to_rotation_matrix(angle, degrees=True))  # apollo
    r2 = R.from_dcm(euler_to_rot(*kaggle_angle, degrees=True))  # kaggle
    assert np.allclose(r1.as_dcm(), r2.as_dcm()), '{}\n{}'.format(r1.as_dcm(), r2.as_dcm())

def generate_kaggle_train_csv(pose_dir, image_dir, save_path):
    car_pose_files = glob.glob(f'{pose_dir}/*json')
    image_files = glob.glob(f'{image_dir}/*jpg')
    dict_pose = {os.path.basename(x).split('.')[0]:x for x in car_pose_files}
    dict_image = {os.path.basename(x).split('.')[0]:x for x in image_files}
    keys_common = list(set(dict_pose.keys()) & set(dict_image.keys()))

    lines = ['ImageId,PredictionString']  # header
    for image_name in tqdm(keys_common):
        car_pose_file = dict_pose[image_name]
        with open(car_pose_file) as f:
            car_poses = json.load(f)
        kaggle_poses = []
        line = '{},'.format(image_name)
        for car_pose in car_poses:
            angle = car_pose['pose'][:3]
            position = car_pose['pose'][3:]
            yaw, pitch, roll = convert_apollo_angle_to_kaggle(angle)
            kaggle_angle = pitch, yaw, roll
            kaggle_pose = [car_pose['car_id']] + list(kaggle_angle) + list(position)
            kaggle_poses.append(kaggle_pose)
        line += ' '.join([str(x) for kaggle_pose in kaggle_poses for x in kaggle_pose])
        lines.append(line)
        # format: "ID_1d7bc9b31,0.5 0.5 0.5 0.0 0.0 0.0 1.0"
    with open(save_path, 'w') as f_out:
        for line in lines:
            f_out.write(line + '\n')


if __name__ == '__main__':
    test_convert_apollo_to_kaggle()

    ## batch conversion
    data_dir = '/Users/pliu/Downloads/3d_car_instance_sample/'
    pose_dir = f'{data_dir}/car_poses'
    image_dir = f'{data_dir}/images'
    save_path = f'{data_dir}/train.csv'
    generate_kaggle_train_csv(pose_dir, image_dir, save_path)