# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Functions to display events and boxes
"""

from __future__ import print_function
import numpy as np
import cv2

LABELMAP = ["car", "pedestrian"]
LABELMAP_LARGE = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']


def make_binary_histo(events, img=None, width=304, height=240):
    """
        simple display function that shows negative events as blue dots and positive as red one
        on a black background
        args :
            - events structured numpy array
            - img (numpy array, height x width x 3) optional array to paint event on.
            - width int
            - height int
        return:
            - img numpy array, height x width x 3)
        """
    background = 'grey'
    if background == 'black':
        if img is None:
            img = 0 * np.ones((height, width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 0
        if events.size:
            assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
            assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)
            # if events['p'] is 1, the image is red, if it is 0, the image is blue
            img[events['y'], events['x'], 2] = 255 * events['p']
            img[events['y'], events['x'], 0] = 255 * (1 - events['p'])

    elif background == 'white':
        if img is None:
            img = 255 * np.ones((height, width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 255
        if events.size:
            assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
            assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)
            # if events['p'] is 1, the image is red, if it is 0, the image is blue
            img[events['y'], events['x'], 2] = 255 * events['p']
            img[events['y'], events['x'], 0] = 255 * (1 - events['p'])
            img[events['y'], events['x'], 1] = 0


    elif background == 'grey':
        if img is None:
            img = 230 * np.ones((height, width, 3), dtype=np.uint8)
        else:
            # if an array was already allocated just paint it grey
            img[...] = 230
        if events.size:
            assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
            assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)
            # if events['p'] is 1, the image is red, if it is 0, the image is blue
            img[events['y'], events['x'], 2] = 255 * events['p']
            img[events['y'], events['x'], 0] = 255 * (1 - events['p'])
            img[events['y'], events['x'], 1] = 0

    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP, colors=None):
    """
    draw bboxes in the image img
    """
    if colors is None:
        colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        colors = [tuple(*item) for item in colors.tolist()]
    else:  # specify the color by the user, input is a tuple uint8, like (128,128,128)
        colors = [colors]*255



    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)



# the target is processed by `parse_label_data_yolo ` and convert to pixel coordinates
def draw_bboxes_target(img, bboxes, labelmap=LABELMAP, linewidth = 1, colors=None):
    # input targets [class, cx, cy, w, h]
    import torch
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if colors is None:
        colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        colors = [tuple(*item) for item in colors.tolist()]
    else:  # specify the color by the user, input is a tuple uint8, like (128,128,128)
        colors = [colors]*255

    for i in range(bboxes.shape[0]):
        center = (int(bboxes[i][1]), int(bboxes[i][2]))
        pt1 = (int(center[0] - bboxes[i][3]//2), int(center[1] - bboxes[i][4]//2))
        pt2 = (int(center[0] + bboxes[i][3]//2), int(center[1] + bboxes[i][4]//2))
        class_id = int(bboxes[i][0])
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        cv2.rectangle(img, pt1, pt2, color, linewidth)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        # cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def draw_bboxes_preds(img, bboxes, labelmap=LABELMAP, linewidth = 1, colors=None):
    # input preds (n,6) tensor per image [xyxy, conf, cls]
    import torch
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if colors is None:
        colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        colors = [tuple(*item) for item in colors.tolist()]
    else:  # specify the color by the user, input is a tuple uint8, like (128,128,128)
        colors = [colors]*255

    for i in range (bboxes.shape[0]):
        pt1 = (int(bboxes[i][0]), int(bboxes[i][1]))
        pt2 = (int(bboxes[i][2]), int(bboxes[i][3]))
        score = bboxes[i][4]
        class_id = int(bboxes[i][5])
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, linewidth)
        cv2.putText(img, class_name, (center[0], pt2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)