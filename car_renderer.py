import os
import cv2
import glob
import pandas as pd
import json
import numpy as np
from matplotlib import pylab as plt

from utils import get_avg_size, draw_obj, euler_to_rot, get_intrinsics
from car_models import car_id2name


class Visualizer(object):
    def __init__(self, basedir, image_dir, model_dir, save_dir=None, ext='.jpg'):
        self.basedir = basedir
        self.image_dir = image_dir
        self.model_dir = model_dir
        self.model_files = self.get_model_files()
        self.anno_df = pd.read_csv(f'{basedir}/train.csv')
        self.avg_size_dict = get_avg_size()
        self.save_dir = save_dir
        self.ext = ext
        self.k = get_intrinsics()

    def get_model_files(self):
        model_files = glob.glob(os.path.join(self.basedir, 'car_models_json/*json'))
        return model_files

    def load_model(self, car_type, type='json'):
        if type != 'json':
            assert TypeError
        model_file = f'{self.model_dir}/{car_type}.json'
        with open(model_file) as json_file:
            data = json.load(json_file)
        vertices = np.array(data['vertices'])
        vertices[:, 1] = -vertices[:, 1]  ## y is pointing downward
        triangles = np.array(data['faces']) - 1
        return vertices, triangles

    def load_model_with_model_id_or_model_class(self, model_id=None, model_class=None):
        assert model_id or model_class
        if model_id:
            car_type = car_id2name[int(model_id)].name
        else:
            car_type = self.avg_size_dict[model_class]['model']
        vertices, triangles = self.load_model(car_type)
        return vertices, triangles

    def load_anno(self, image_name):
        pred_string = self.anno_df[self.anno_df.ImageId == image_name].PredictionString.iloc[0]
        items = pred_string.split(' ')
        model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
        return model_types, yaws, pitches, rolls, xs, ys, zs

    def load_image(self, image_name):
        img = cv2.imread(f'{self.image_dir}/{image_name}{self.ext}',
                         cv2.COLOR_BGR2RGB)[:, :, ::-1]
        H, W = img.shape[:2]
        print('image size is H={}, W={}'.format(H, W))
        return img

    def render_mask(self, yaw, pitch, roll, x, y, z, k=None,
                    overlay=False, img=None, mask_color=(0, 0, 255),
                    vertices=None, triangles=None,
                    vis_bbox=False, bbox_color=(0, 255, 0)):

        yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
        # I think the pitch and yaw should be exchanged
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T
        img_cor_points = np.dot(k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        # get amodal bbox
        xmin, ymin = img_cor_points[:, :2].min(axis=0).astype(int)
        xmax, ymax = img_cor_points[:, :2].max(axis=0).astype(int)
        # NB. Do not truncate to get amodal bbox
        # truncate bbox to image boundary
        # H, W = img.shape[:2]
        # xmin = max(0, xmin)
        # ymin = max(0, ymin)
        # xmax = min(xmax, W - 1)
        # ymax = min(ymax, H - 1)

        if not overlay:
            canvas = np.zeros_like(img)
        else:
            canvas = img  #.copy()  # sometimes it needs to have copy to work
        canvas = draw_obj(canvas, img_cor_points, triangles, color=mask_color)
        if overlay:
            alpha = .5
            img = np.array(img)
            cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, img)
            if vis_bbox:
                canvas = cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), bbox_color, 10)

        return canvas, (xmin, ymin, xmax, ymax)

    def render_mask_v2(self, yaw, pitch, roll, x, y, z, k=None,
                    overlay=False, img=None, mask_color=(0, 0, 255),
                    vertices=None, triangles=None,
                    vis_bbox=False, bbox_color=(0, 255, 0)):

        yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
        # note the order change
        roll, pitch, yaw = pitch, yaw, roll
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_rot(roll, pitch, yaw)
        Rt = Rt[:3, :]
        self.rotmat = Rt[:3, :3]
        print(self.rotmat, self.rotmat.shape)
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T
        img_cor_points = np.dot(k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        # get amodal bbox
        xmin, ymin = img_cor_points[:, :2].min(axis=0).astype(int)
        xmax, ymax = img_cor_points[:, :2].max(axis=0).astype(int)
        # NB. Do not truncate to get amodal bbox
        # truncate bbox to image boundary
        # H, W = img.shape[:2]
        # xmin = max(0, xmin)
        # ymin = max(0, ymin)
        # xmax = min(xmax, W - 1)
        # ymax = min(ymax, H - 1)

        if not overlay:
            canvas = np.zeros_like(img)
        else:
            canvas = img  # .copy()  # sometimes it needs to have copy to work
        img_cor_points = np.expand_dims(img_cor_points[:, :2], axis=1)

        # draw_obj func in apolloScape api
        for face in triangles:
            pts = np.array([[img_cor_points[idx, 0, 0], img_cor_points[idx, 0, 1]] for idx in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, mask_color)

        if overlay:
            alpha = .5
            img = np.array(img)
            cv2.addWeighted(canvas, alpha, img, 1 - alpha, 0, img)
            if vis_bbox:
                canvas = cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), bbox_color, 10)

        return canvas, (xmin, ymin, xmax, ymax)

    def process(self, image_name):
        img = self.load_image(image_name)
        model_types, yaws, pitches, rolls, xs, ys, zs = self.load_anno(image_name)
        bboxes = []
        overlay = img.copy()
        for yaw, pitch, roll, x, y, z, model_id in zip(yaws, pitches, rolls, xs, ys, zs, model_types):
            vertices, triangles = self.load_model_with_model_id_or_model_class(model_id=model_id)
            overlay, bbox = self.render_mask(yaw, pitch, roll, x, y, z, k=self.k,
                                             overlay=True, img=overlay, mask_color=(0, 0, 255),
                                             vertices=vertices, triangles=triangles,
                                             vis_bbox=True, bbox_color=(0, 255, 0))
            bboxes.append(bbox)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'{image_name}{self.ext}')
            print(f'saving vis to {save_path}')
            plt.imsave(save_path, overlay)
        else:
            plt.figure(figsize=(20, 20))
            plt.imshow(overlay)
            plt.show()
        return overlay, bboxes


if __name__ == '__main__':
    basedir = '/Users/pliu/Downloads/pku_kaggle_data/'
    image_dir = '/Users/pliu/Downloads/pku_kaggle_data/train_images'
    model_dir = '/Users/pliu/Downloads/pku_kaggle_data/car_models_json'
    save_dir = '/Users/pliu/Downloads/pku_kaggle_data/vis_anno/'
    # save_dir = None  # show in terminal or notebook

    image_name = 'ID_001d6829a'
    vis = Visualizer(basedir=basedir, image_dir=image_dir, model_dir=model_dir, save_dir=save_dir)
    overlay, bboxes = vis.process(image_name=image_name)