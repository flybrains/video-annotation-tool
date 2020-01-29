import vat_core as vc
import pickle
import numpy as np
import cv2

def get_patch_info(patch, x,y):
    if patch[int(x),int(y)]==255:
        status = True
    else:
        status = False
    patch = (255-patch)
    contours, _ = cv2.findContours(patch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c)<(0.75*patch.shape[0]*patch.shape[1])]
    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = np.float16(M['m10']/M['m00'])
            cy = np.float16(M['m01']/M['m00'])
            centroids.append(np.asarray([int(cx),int(cy)]))
    hinge = np.asarray([int(x), int(y)])
    dists = [np.linalg.norm(hinge-c) for c in centroids]
    dist = min(dists)

    return status, dist


def get_sorted_dists(frame_info, target_id, longest_tuple):

    sorted_dists = []
    sorted_ids = []

    target = frame_info.list_of_contour_points[target_id-1]
    hinge_point = np.asarray([int(target.x), int(target.y)])
    all_but = [pt for pt in frame_info.list_of_contour_points if pt.id!=target_id]

    object_dist_tuples = []

    for pt in all_but:
        to_calc = hinge_point - np.asarray([int(pt.x), int(pt.y)])
        dist = np.linalg.norm(to_calc)
        odt = (pt.id, dist)
        object_dist_tuples.append(odt)

    s_object_dist_tuples = sorted(object_dist_tuples, key=lambda d: d[1])
    sorted_dists = [e[1] for e in s_object_dist_tuples]
    sorted_ids = [e[0] for e in s_object_dist_tuples]

    n_to_add = longest_tuple - len(sorted_ids)
    if n_to_add>1:
        for i in range(n_to_add):
            sorted_dists.append(None)
            sorted_ids.append(None)

    return sorted_dists, sorted_ids

def scan_for_longest_dist_tuple(videoinfo):
    return max([len(e.list_of_contour_points) for e in videoinfo.get_frame_list()])

def make_rows(videoinfo):
    max_len = scan_for_longest_dist_tuple(videoinfo)
    foodpatch_mask = videoinfo.metadata['food_patch_mask']
    rows = []
    foodpatch_mask = cv2.cvtColor(foodpatch_mask, cv2.COLOR_BGR2GRAY)

    for frame_info in videoinfo.get_frame_list():
        print('-----------------')
        frame_idx = frame_info.index
        video_idx = frame_info.frameNo
        for idx, pt in enumerate(frame_info.list_of_contour_points):

            id, x, y = pt.id, pt.x, pt.y
            sex = frame_info.behavior_list[idx][0]
            behavior = frame_info.behavior_list[idx][1]
            row = [frame_idx, video_idx, id, sex, behavior]

            patch_status, dist_to_closest_patch = get_patch_info(foodpatch_mask,x,y)
            sorted_dists, sorted_ids = get_sorted_dists(frame_info, id, max_len)
            row.append(patch_status)
            row.append(dist_to_closest_patch)
            row.append(x)
            row.append(y)
            for i in range(len(sorted_dists)):
                row.append(sorted_dists[i])
                row.append(sorted_ids[i])
        rows.append(row)
    header = ['labelled_frame', 'video_frame', 'animal_id', "sex", "behavior", 'on_patch', 'dist_to_closest_patch_centroid','x_pos','y_pos']

    for i in range(len(sorted_dists)):
        if i==0:
            header.append('dist_to_1st_closest')
            header.append('1st_closest_id')
        elif i==1:
            header.append('dist_to_2nd_closest')
            header.append('2nd_closest_id')
        elif i==2:
            header.append('dist_to_3rd_closest')
            header.append('3rd_closest_id')
        else:
            header.append('dist_to_{}th_closest'.format(i+1))
            header.append('{}th_closest_id'.format(i+1))

    return header, rows

if __name__=='__main__':
    with open('/home/patrick/code/video-annotation-tool/data/sample1/sample1.pkl', 'rb') as f:
        videoinfo = pickle.load(f)
    header,rows = make_rows(videoinfo)









#
#
# import pandas as pd
#
# cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
#         'Price': [22000,25000,27000,35000]
#         }
#
# df = pd.DataFrame(cars, columns = ['Brand', 'Price'])
#
# print (df)
