import numpy as np
import json
import glob
import os
import cv2
import matplotlib.pylab as plt


def gauss(n=11, mean = 0, sigma=1):
    from math import pi, sqrt, exp
    r = (np.arange(n) - mean).astype(int)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def make_kernel(ymin, ymax, n=948, type='box'):
    occ_t = np.zeros(n)
    y_width = ymax - ymin
    y_center = (ymax + ymin) // 2
    if type == 'box':
        occ_t[ymin:ymax] = 1
    elif type == 'gaussian':
        occ_t = gauss(n=n, mean=y_center, sigma=y_width)
    return occ_t


def get_profile(height, box_list, ignore_small=50, type='gaussian'):
    occ = np.zeros(height)
    for box in box_list:
        ymin, ymax = int(box[0]), int(box[2])
        if ymax - ymin < ignore_small:  # orig scale
            continue
        ymin, ymax = int(ymin/2), int(ymax/2)
        occ_t = make_kernel(ymin, ymax, type=type, n=height)

        occ += occ_t
    occ /= occ.max()
    peak_loc = np.argmax(occ)
#     print('max at {} for {}'.format(peak_loc, type))
    return occ, peak_loc


def read_json(json_file):
    with open(json_file, 'r') as f_in:
        data = json.load(f_in)
    return data

def poly2box(poly):
    xs = [pt['x'] for pt in poly]
    ys = [pt['y'] for pt in poly]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    box = [ymin, xmin, ymax, xmax]
    return box

def overlay_box(canvas, box, scale=1):
    box = np.array(box) * scale
    canvas = cv2.rectangle(canvas, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (255, 255, 0), 2)
    return canvas

def get_bbox_on_canvas(img, box):
    canvas = np.zeros_like(img)
    for box in box_list:
        canvas = overlay_box(canvas, box, scale=1/2)
    return canvas


class VanishingPoint(object):
    def __init__(self, basedir, visualize_vp=False, visualize_img=False):
        self.basedir = basedir
        self.visualize_vp = visualize_vp
        self.visualize_img = visualize_img
        self.data_dict = self.load_data(basedir)
        sample_key = list(self.data_dict.keys())[0]
        self.height, self.width = plt.imread(self.data_dict[sample_key]['image']).shape[:2]

    @staticmethod
    def load_data(basedir):
        image_files = glob.glob(os.path.join(basedir, '*png'))
        json_files = glob.glob(os.path.join(basedir, '*json'))
        key_fn = lambda x: os.path.basename(x).split('.')[0]

        image_dict = {key_fn(x): x for x in image_files}
        json_dict = {key_fn(x): x for x in json_files}
        common_keys = list(set(image_dict.keys()) & set(json_dict.keys()))
        data_dict = {x: {'image': image_dict[x], 'json': json_dict[x]} for x in common_keys}
        return data_dict

    def visualize_image(self):
        img = plt.imread(self.image_file)
        plt.imshow(img)
        canvas = get_bbox_on_canvas(img, self.box_list)
        plt.figure()
        plt.imshow(canvas)

    def find_vanishing_line(self):
        occ_gaussian, peak_loc_gaussian = get_profile(self.height, self.box_list, type='gaussian')
        occ_box, peak_loc_box = get_profile(self.height, self.box_list, type='box')
        if self.visualize_vp:
            plt.plot(occ_box, label='box')
            plt.plot(occ_gaussian, label='gaussian')
            plt.legend()
            plt.title('vanishing point at y={}'.format(peak_loc_gaussian))
        return peak_loc_box

    def process(self):
        for key in sorted(list(self.data_dict.keys()))[:2]:
            data = read_json(self.data_dict[key]['json'])
            self.image_file = self.data_dict[key]['image']
            self.box_list = [poly2box(x['poly']) for x in data['objects']]
            peak_loc_box = self.find_vanishing_line()
            print(key, len(self.box_list), peak_loc_box)


if __name__ == '__main__':
    basedir = '/Users/pliu/Downloads/random_300_images/'

    vp = VanishingPoint(basedir=basedir, visualize_img=False, visualize_vp=False)
    vp.process()
