import numpy as np
import json
import glob
import os
import cv2
from tqdm import tqdm
import matplotlib.pylab as plt


def gauss(n=11, mean = 0, sigma=1):
    from math import pi, sqrt, exp
    r = (np.arange(n) - mean).astype(int)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def make_kernel(ymin, ymax, n=948, type='box'):
    occ_t = np.zeros(n)
    y_width = ymax - ymin
    y_center = (ymax + ymin) // 2
    y_center_skewed = int(ymax * (1/4) + ymin * (3/4))
    if type == 'box':
        occ_t[ymin:ymax] = 1
    elif type == 'gaussian':
        occ_t = gauss(n=n, mean=y_center, sigma=y_width)
    elif type == 'skewed_gaussian':
        occ_t = gauss(n=n, mean=y_center_skewed, sigma=y_width/2)
    return occ_t


def get_profile(height, box_list, type='skewed_gaussian'):
    occ = np.zeros(height)
    for box in box_list:
        ymin, ymax = int(box[0]), int(box[2])
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


def write_json(data, json_file):
    with open(json_file, 'w') as f_out:
        data = json.dump(data, f_out, sort_keys=True, indent=4)



def poly2box(poly):
    xs = [pt['x'] for pt in poly]
    ys = [pt['y'] for pt in poly]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    box = [ymin, xmin, ymax, xmax]
    return box


def get_bbox_on_canvas(img, box_list, color=(255, 255, 0), use_empty_canvas=True):
    if use_empty_canvas:
        height, width = img.shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = img
    for box in box_list:
        canvas = cv2.rectangle(
            canvas, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), color, 10)
    return canvas


def filter_box_list(box_list, ignore_small=50):
    new_box_list = []
    for box in box_list:
        ymin, ymax = int(box[0]), int(box[2])
        # print(ymax - ymin)
        if ymax - ymin < ignore_small:  # orig scale
            continue
        new_box_list.append(box)
    # print(len(box_list), len(new_box_list))
    return new_box_list


def visualize_image(image_file, box_list, image_scale=2, y_vp=None,
                    vp_color=(0, 255, 0), box_color=(255, 255, 0)):
    """

    Args:
        image_file:
        box_list:
        image_scale:
        y_vp: y coordinate of the vanishing point/line

    Returns:

    """
    img = plt.imread(image_file, -1)
    img = cv2.resize(img, (0, 0), fx=image_scale, fy=image_scale)
    width = img.shape[1]
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[..., :3].astype(np.uint8)
    if y_vp is not None:
        img = cv2.line(img, (0, y_vp), (width - 1, y_vp), color=vp_color, thickness=10)
    plt.figure()
    plt.imshow(img)

    img = get_bbox_on_canvas(img, box_list, color=box_color)
    if y_vp is not None:
        img = cv2.line(img, (0, y_vp), (width - 1, y_vp), color=vp_color, thickness=10)
    plt.figure()
    plt.imshow(img)


def visualize_profile(height, box_list):
    occ_gaussian, peak_loc_gaussian = get_profile(height, box_list, type='skewed_gaussian')
    occ_box, peak_loc_box = get_profile(height, box_list, type='box')
    plt.plot(occ_box, label='box')
    plt.plot(occ_gaussian, label='skewed_gaussian')
    plt.legend()
    plt.title('vanishing point at y={}'.format(peak_loc_gaussian))
    print('box', peak_loc_box, 'gauss', peak_loc_gaussian)
    return peak_loc_gaussian


class VanishingPoint(object):
    def __init__(self, basedir, image_scale=2, visualize_vp=False, visualize_img=False):
        self.basedir = basedir
        self.image_scale = image_scale
        self.visualize_vp = visualize_vp
        self.visualize_img = visualize_img
        self.data_dict = self.load_data(basedir)
        sample_key = list(self.data_dict.keys())[0]
        image_shape = plt.imread(self.data_dict[sample_key]['image']).shape[:2]
        self.height, self.width = (np.array(image_shape) * self.image_scale).astype(int)
        self.box_list_all = []
        self.box_list_all_valid = []

    @staticmethod
    def load_data(basedir):
        image_files = glob.glob(os.path.join(basedir, '*png'))
        json_files = glob.glob(os.path.join(basedir, '*json'))
        key_fn = lambda x: os.path.basename(x).split('.')[0]

        image_dict = {key_fn(x): x for x in image_files}
        json_dict = {key_fn(x): x for x in json_files}
        common_keys = list(set(image_dict.keys()) & set(json_dict.keys()))
        data_dict = {x: {'image': image_dict[x], 'json': json_dict[x]} for x in common_keys}
        is_data_enough = (len(data_dict) >= 20)
        if not is_data_enough:
            print('{} has only {} images'.format(os.path.basename(basedir), len(data_dict)))
        return data_dict

    def find_vanishing_line_per_image(self):
        occ_gaussian, peak_loc_gaussian = get_profile(self.height, self.box_list, type='skewed_gaussian')
        occ_box, peak_loc_box = get_profile(self.height, self.box_list, type='box')
        if self.visualize_vp:
            plt.plot(occ_box, label='box')
            plt.plot(occ_gaussian, label='skewed_gaussian')
            plt.legend()
            plt.title('vanishing point at y={}'.format(peak_loc_gaussian))
        return peak_loc_gaussian

    def visual_check(self, y_vp=0, inject_y_list=[]):
        """Visual check after process

        Args:
            inject_y_list: when not empty, display them as well

        Returns:

        """
        for key in sorted(list(self.data_dict.keys()))[:]:
            image_file = self.data_dict[key]['image']
            img = plt.imread(image_file)[..., :3]
            img = cv2.resize(img, (0, 0), fx=2, fy=2)
            img = cv2.line(img, (0, y_vp), (self.width - 1, y_vp), color=(0, 255, 0), thickness=10)
            for y in inject_y_list:
                img = cv2.line(img, (0, y), (self.width-1, y_vp), color=(0, 255, 0), thickness=10)
            plt.figure()
            plt.imshow(img)
            loc_key = os.path.basename(self.basedir)
            plt.title(loc_key + ': ' + key)

    def process(self):
        vp_dict = {}
        for key in sorted(list(self.data_dict.keys()))[:]:
            data = read_json(self.data_dict[key]['json'])
            if isinstance(data, dict):
                data = data['objects']
            elif isinstance(data, list):
                data = data
            else:
                raise TypeError
            image_file = self.data_dict[key]['image']
            # do NOT use truncated bbox
            try:
                box_list = [poly2box(x['poly']) for x in data if (not x['MOC']['cropped'] and x['poly'])]
            except:
                print(key)
                continue
            box_list = filter_box_list(box_list)
            self.box_list_all.append(box_list)
            if len(box_list) < 4:
                continue
            self.box_list_all_valid.append(box_list)

            occ_gaussian, peak_loc_gaussian = get_profile(self.height, box_list, type='skewed_gaussian')
            vp_dict[key] = peak_loc_gaussian
        # for all images
        self.box_list_all_flat = [y for x in self.box_list_all for y in x]  # flatten
        occ_gaussian, peak_loc_gaussian = get_profile(
            self.height, self.box_list_all_flat, type='skewed_gaussian')
        vp_dict['all'] = peak_loc_gaussian
        return vp_dict


def dump_dict(db_dict, save_dir=None):
    """

    Args:
        db_dict: the dictionary containing one vanishing point per car-city-time
        savedir: if None, do not save

    Returns:

    """
    db_dict = {k: int(v) for k, v in db_dict.items()}
    # flip loc and time: from car-city-time to car-time-city
    new_db_dict = {}
    for k in db_dict.keys():
        v0, v1, v2 = k.split('-')
        new_k = '-'.join([v0, v2, v1])
        new_db_dict[new_k] = int(db_dict[k])
    if save_dir:
        write_json(db_dict, os.path.join(save_dir, 'car-city-time.json'))
        write_json(new_db_dict, os.path.join(save_dir, 'car-time-city.json'))


def batch_find_vp(datadir, save_dir=None):
    """

    Args:
        datadir: contains a list of dir, each dir in the format of car-city-time
        save_dir: if not None, save to save_dir

    Returns:

    """
    basedir_list = glob.glob(os.path.join(datadir, '*'))
    vp_dict_all = {}
    for basedir in tqdm(basedir_list[:]):
        vp = VanishingPoint(basedir=basedir, visualize_img=False, visualize_vp=False)
        vp_dict_per_image = vp.process()

        # collect all
        loc_key = os.path.basename(basedir)
        vp_dict_all[loc_key] = vp_dict_per_image['all']
    dump_dict(vp_dict_all, save_dir=save_dir)
    return vp_dict_all


if __name__ == '__main__':
    basedir = '/Users/pliu/Downloads/random_300_images/'
    vp = VanishingPoint(basedir=basedir, visualize_img=False, visualize_vp=False)
    vp_dict = vp.process()
    # print(vp_dict)

    datadir = '/Users/pliu/Downloads/sample_img_and_label/'
    vp_dict_all = batch_find_vp(datadir)
    print(vp_dict_all)