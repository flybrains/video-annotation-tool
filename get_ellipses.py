import os
import sys
import cv2
import pickle
import csv
import numpy as np
import vat_core as vc
from skimage.measure import EllipseModel


def _detect_ellipses(mat):
    mat = (255-mat)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c)<(0.75*mat.shape[0]*mat.shape[1])]

    ellipses = []

    for c in contours:
        m = cv2.fitEllipse(contours[0])
        mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
        mat = cv2.ellipse(mat, m, (0,0,255))
        ellipses.append(m)
        # cv2.imshow('m', mat)
        # cv2.waitKey(0)
    return ellipses




def load_pickle(pickle_name):
    with open(pickle_name, 'rb') as f:
        videoinfo = pickle.load(f)
    return videoinfo

def get_dish_ellipse(videoinfo):
    ellipses = _detect_ellipses(videoinfo.metadata['bowl_mask'])
    return ellipses

def _get_dish_width(bowl_mask):
    max_spread = 0
    for row in range(int(bowl_mask.shape[0]/2)):
        try:
            row_only = bowl_mask[row*2,:]
            where_white = list(np.greater(row_only, 0))
            first = where_white.index(True)
            f_where_white =  [e for e in reversed(where_white)]
            last = len(where_white) - f_where_white.index(True)
            spread = last - first
            if spread > max_spread:
                max_spread = spread
        except ValueError:
            pass
    dish_width = max_spread
    return dish_width

def get_food_ellipses(videoinfo):
    ellipses = _detect_ellipses(videoinfo.metadata['food_patch_mask'])
    return ellipses

def write_to_file(pickle_name, dish_info, food_infos, videoinfo):


    diameter_mm = videoinfo.metadata['chamber_d']
    bowl_mask = cv2.cvtColor(videoinfo.metadata['bowl_mask'], cv2.COLOR_BGR2GRAY)
    dish_width = _get_dish_width(bowl_mask)
    cf = float(diameter_mm/dish_width)

    address = pickle_name.split('.')[0]+'_ellipses.txt'
    with open(address, 'w') as f:
        f.write('dish ellipse : pixel_distances (x, y, minor_axis, major_axis, angle)\n')
        f.write('{},{},{},{},{}\n\n'.format(dish_info[0][0][0], dish_info[0][0][1],dish_info[0][1][0],dish_info[0][1][1],dish_info[0][2]))

        for idx, info in enumerate(food_infos):
            f.write('food ellipse {}: pixel_distances (x, y, minor_axis, major_axis, angle)\n'.format(idx+1))
            f.write('{},{},{},{},{}\n\n'.format(dish_info[0][0][0], dish_info[0][0][1],dish_info[0][1][0],dish_info[0][1][1],dish_info[0][2]))

        f.write('dish ellipse : converted_mm_distances (x, y, minor_axis, major_axis, angle)\n')
        f.write('{},{},{},{},{}\n\n'.format(dish_info[0][0][0]*cf, dish_info[0][0][1]*cf,dish_info[0][1][0]*cf,dish_info[0][1][1]*cf,dish_info[0][2]))

        for idx, info in enumerate(food_infos):
            f.write('food ellipse {}: converted_mm_distances (x, y, minor_axis, major_axis, angle)\n'.format(idx+1))
            f.write('{},{},{},{},{}\n\n'.format(dish_info[0][0][0]*cf, dish_info[0][0][1]*cf,dish_info[0][1][0]*cf,dish_info[0][1][1]*cf,dish_info[0][2]))




if __name__=="__main__":
    name = sys.argv[1].split('/')[-1]
    pickle_name = os.path.join(sys.argv[1], '{}.pkl'.format(name))
    videoinfo = load_pickle(pickle_name)
    dish_info = get_dish_ellipse(videoinfo)
    food_info = get_food_ellipses(videoinfo)
    write_to_file(pickle_name, dish_info, food_info, videoinfo)
