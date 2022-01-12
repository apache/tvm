# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument, no-init,
"""
Yolo detection boxes helper functions
====================
DarkNet helper functions for yolo and image loading.
This functions will not be loaded by default.
These are utility functions used for testing and tutorial file.
"""
from __future__ import division
import math
from collections import namedtuple
from functools import cmp_to_key
import numpy as np

Box = namedtuple("Box", ["x", "y", "w", "h"])


def nms_comparator(a, b):
    if "sort_class" in b and b["sort_class"] >= 0:
        diff = a["prob"][b["sort_class"]] - b["prob"][b["sort_class"]]
    else:
        diff = a["objectness"] - b["objectness"]
    return diff


def _correct_boxes(dets, w, h, netw, neth, relative):
    new_w, new_h = (netw, (h * netw) // w) if (netw / w < neth / h) else ((w * neth // h), neth)
    for det in dets:
        b = det["bbox"]
        b = b._replace(x=(b.x - (netw - new_w) / 2 / netw) / (new_w / netw))
        b = b._replace(y=(b.y - (neth - new_h) / 2 / neth) / (new_h / neth))
        b = b._replace(w=b.w * netw / new_w)
        b = b._replace(h=b.h * neth / new_h)
        if not relative:
            b = b._replace(x=b.x * w)
            b = b._replace(w=b.w * w)
            b = b._replace(y=b.y * h)
            b = b._replace(h=b.h * h)
        det["bbox"] = b
    return dets


def _overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r1 if r1 < r2 else r2
    return right - left


def _box_intersection(a, b):
    w = _overlap(a.x, a.w, b.x, b.w)
    h = _overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    return w * h


def _box_union(a, b):
    i = _box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def _box_iou(a, b):
    return _box_intersection(a, b) / _box_union(a, b)


def _get_box(data, biases, n, location, lw, lh, w, h):
    bx = (location[2] + data[location[0]][0][location[1]][location[2]]) / lw
    by = (location[1] + data[location[0]][1][location[1]][location[2]]) / lh
    bw = np.exp(data[location[0]][2][location[1]][location[2]]) * biases[2 * n] / w
    bh = np.exp(data[location[0]][3][location[1]][location[2]]) * biases[2 * n + 1] / h
    return Box(bx, by, bw, bh)


def _get_yolo_detections(l, im_shape, net_shape, thresh, relative, dets):
    data = l["output"]
    active_data_loc = np.asarray(np.where(data[:, 4, :, :] > thresh))
    before_correct_dets = []
    for i in range(active_data_loc.shape[1]):
        location = [active_data_loc[0][i], active_data_loc[1][i], active_data_loc[2][i]]
        box_b = _get_box(
            data,
            l["biases"],
            np.asarray(l["mask"])[location[0]],
            location,
            data.shape[3],
            data.shape[2],
            net_shape[0],
            net_shape[1],
        )
        objectness = data[location[0]][4][location[1]][location[2]]
        classes = l["classes"]
        prob = objectness * data[location[0], 5 : 5 + 1 + classes, location[1], location[2]]
        prob[prob < thresh] = 0
        detection = {}
        detection["bbox"] = box_b
        detection["classes"] = classes
        detection["prob"] = prob
        detection["objectness"] = objectness
        before_correct_dets.append(detection)
    dets.extend(
        _correct_boxes(
            before_correct_dets, im_shape[0], im_shape[1], net_shape[0], net_shape[1], relative
        )
    )


def _get_region_detections(l, im_shape, net_shape, thresh, relative, dets):
    data = l["output"]
    before_correct_dets = []
    for row in range(data.shape[2]):
        for col in range(data.shape[3]):
            for n in range(data.shape[0]):
                prob = [0] * l["classes"]
                scale = data[n, l["coords"], row, col] if not l["background"] else 1
                location = [n, row, col]
                box_b = _get_box(
                    data,
                    l["biases"],
                    n,
                    location,
                    data.shape[3],
                    data.shape[2],
                    data.shape[3],
                    data.shape[2],
                )
                objectness = scale if scale > thresh else 0
                if objectness:
                    prob = (
                        scale * data[n, l["coords"] + 1 : l["coords"] + 1 + l["classes"], row, col]
                    )
                    prob[prob < thresh] = 0
                detection = {}
                detection["bbox"] = box_b
                detection["prob"] = prob
                detection["objectness"] = objectness
                before_correct_dets.append(detection)
    _correct_boxes(
        before_correct_dets, im_shape[0], im_shape[1], net_shape[0], net_shape[1], relative
    )
    dets.extend(before_correct_dets)


def fill_network_boxes(net_shape, im_shape, thresh, relative, tvm_out):
    dets = []
    for layer in tvm_out:
        if layer["type"] == "Yolo":
            _get_yolo_detections(layer, im_shape, net_shape, thresh, relative, dets)
        elif layer["type"] == "Region":
            _get_region_detections(layer, im_shape, net_shape, thresh, relative, dets)
    return dets


def do_nms_sort(dets, classes, thresh):
    "Does the sorting based on the threshold values"
    k = len(dets) - 1
    cnt = 0
    while cnt < k:
        if dets[cnt]["objectness"] == 0:
            dets[k], dets[cnt] = dets[cnt], dets[k]
            k = k - 1
        else:
            cnt = cnt + 1
    total = k + 1
    for k in range(classes):
        for i in range(total):
            dets[i]["sort_class"] = k
        dets[0:total] = sorted(dets[0:total], key=cmp_to_key(nms_comparator), reverse=True)
        for i in range(total):
            if dets[i]["prob"][k] == 0:
                continue
            a = dets[i]["bbox"]
            for j in range(i + 1, total):
                b = dets[j]["bbox"]
                if _box_iou(a, b) > thresh:
                    dets[j]["prob"][k] = 0


def get_detections(im, det, thresh, names, classes):
    "Draw the markings around the detected region"
    labelstr = []
    category = -1
    detection = None
    valid = False
    for j in range(classes):
        if det["prob"][j] > thresh:
            if category == -1:
                category = j
            labelstr.append(names[j] + " " + str(round(det["prob"][j], 4)))

    if category > -1:
        valid = True
        imc, imh, imw = im.shape
        width = int(imh * 0.006)
        offset = category * 123457 % classes
        red = _get_color(2, offset, classes)
        green = _get_color(1, offset, classes)
        blue = _get_color(0, offset, classes)
        rgb = [red, green, blue]
        b = det["bbox"]
        left = int((b.x - b.w / 2.0) * imw)
        right = int((b.x + b.w / 2.0) * imw)
        top = int((b.y - b.h / 2.0) * imh)
        bot = int((b.y + b.h / 2.0) * imh)

        if left < 0:
            left = 0
        if right > imw - 1:
            right = imw - 1
        if top < 0:
            top = 0
        if bot > imh - 1:
            bot = imh - 1

        detection = {
            "category": category,
            "labelstr": labelstr,
            "left": left,
            "top": top,
            "right": right,
            "bot": bot,
            "width": width,
            "rgb": rgb,
        }

    return valid, detection


def draw_detections(font_path, im, dets, thresh, names, classes):
    "Draw the markings around the detected region"
    for det in dets:
        valid, detection = get_detections(im, det, thresh, names, classes)
        if valid:
            rgb = detection["rgb"]
            label = _get_label(font_path, "".join(detection["labelstr"]), rgb)
            _draw_box_width(
                im,
                detection["left"],
                detection["top"],
                detection["right"],
                detection["bot"],
                detection["width"],
                rgb[0],
                rgb[1],
                rgb[2],
            )
            _draw_label(im, detection["top"] + detection["width"], detection["left"], label, rgb)


def show_detections(im, dets, thresh, names, classes):
    "Print the markings and the detected region"
    for det in dets:
        valid, detection = get_detections(im, det, thresh, names, classes)
        if valid:
            print(
                "class:{} left:{} right:{} top:{} bottom:{}".format(
                    detection["labelstr"],
                    detection["left"],
                    detection["top"],
                    detection["right"],
                    detection["bot"],
                )
            )


def _get_pixel(im, x, y, c):
    return im[c][y][x]


def _set_pixel(im, x, y, c, val):
    if x < 0 or y < 0 or c < 0 or x >= im.shape[2] or y >= im.shape[1] or c >= im.shape[0]:
        return
    im[c][y][x] = val


def _draw_label(im, r, c, label, rgb):
    w = label.shape[2]
    h = label.shape[1]
    if (r - h) >= 0:
        r = r - h

    for j in range(h):
        if j < h and (j + r) < im.shape[1]:
            for i in range(w):
                if i < w and (i + c) < im.shape[2]:
                    for k in range(label.shape[0]):
                        val = _get_pixel(label, i, j, k)
                        _set_pixel(im, i + c, j + r, k, val)  # rgb[k] * val)


def _get_label(font_path, labelstr, rgb):
    # pylint: disable=import-outside-toplevel
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    text = labelstr
    colorText = "black"
    testDraw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = ImageFont.truetype(font_path, 25)
    width, height = testDraw.textsize(labelstr, font=font)
    img = Image.new(
        "RGB", (width, height), color=(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    )
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, fill=colorText, font=font)
    opencvImage = np.divide(np.asarray(img), 255)
    return opencvImage.transpose(2, 0, 1)


def _get_color(c, x, max_value):
    c = int(c)
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (float(x) / float(max_value)) * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio -= i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return r


def _draw_box(im, x1, y1, x2, y2, r, g, b):
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    ac, ah, aw = im.shape
    if x1 < 0:
        x1 = 0
    if x1 >= aw:
        y1 = 0
    if y1 >= ah:
        y1 = ah - 1
    if y2 < 0:
        y2 = 0
    if y2 >= ah:
        y2 = ah - 1

    for i in range(x1, x2):
        im[0][y1][i] = r
        im[0][y2][i] = r
        im[1][y1][i] = g
        im[1][y2][i] = g
        im[2][y1][i] = b
        im[2][y2][i] = b

    for i in range(y1, y2):
        im[0][i][x1] = r
        im[0][i][x2] = r
        im[1][i][x1] = g
        im[1][i][x2] = g
        im[2][i][x1] = b
        im[2][i][x2] = b


def _draw_box_width(im, x1, y1, x2, y2, w, r, g, b):
    for i in range(int(w)):
        _draw_box(im, x1 + i, y1 + i, x2 - i, y2 - i, r, g, b)
