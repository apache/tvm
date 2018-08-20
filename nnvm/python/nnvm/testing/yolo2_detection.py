# pylint: disable=invalid-name, unused-variable, unused-argument, no-init
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
import numpy as np

def _entry_index(batch, w, h, outputs, classes, coords, location, entry):
    n = int(location/(w*h))
    loc = location%(w*h)
    return batch*outputs + n*w*h*(coords+classes+1) + entry*w*h + loc

Box = namedtuple('Box', ['x', 'y', 'w', 'h'])
def _get_region_box(x, biases, n, index, i, j, w, h, stride):
    b = Box(0, 0, 0, 0)
    b = b._replace(x=(i + x[index + 0*stride]) / w)
    b = b._replace(y=(j + x[index + 1*stride]) / h)
    b = b._replace(w=np.exp(x[index + 2*stride]) * biases[2*n] / w)
    b = b._replace(h=np.exp(x[index + 3*stride]) * biases[2*n+1] / h)
    return b

def _correct_region_boxes(boxes, n, w, h, netw, neth, relative):
    new_w, new_h = (netw, (h*netw)/w) if (netw/w < neth/h) else ((w*neth/h), neth)
    for i in range(n):
        b = boxes[i]
        b = boxes[i]
        b = b._replace(x=(b.x - (netw - new_w)/2/netw) / (new_w/netw))
        b = b._replace(y=(b.y - (neth - new_h)/2/neth) / (new_h/neth))
        b = b._replace(w=b.w * netw/new_w)
        b = b._replace(h=b.h * neth/new_h)
        if not relative:
            b = b._replace(x=b.x * w)
            b = b._replace(w=b.w * w)
            b = b._replace(y=b.y * h)
            b = b._replace(h=b.h * h)
        boxes[i] = b

def _overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 < r2 else r2
    return right - left

def _box_intersection(a, b):
    w = _overlap(a.x, a.w, b.x, b.w)
    h = _overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    return w*h

def _box_union(a, b):
    i = _box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u

def _box_iou(a, b):
    return _box_intersection(a, b)/_box_union(a, b)

def get_region_boxes(layer_in, imw, imh, netw, neth, thresh, probs,
                     boxes, relative, tvm_out):
    "To get the boxes for the image based on the prediction"
    lw = layer_in.w
    lh = layer_in.h
    probs = [[0 for i in range(layer_in.classes + 1)] for y in range(lw*lh*layer_in.n)]
    boxes = [Box(0, 0, 0, 0) for i in range(lw*lh*layer_in.n)]
    for i in range(lw*lh):
        row = int(i / lw)
        col = int(i % lw)
        for n in range(layer_in.n):
            index = n*lw*lh + i
            obj_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                     layer_in.coords, n*lw*lh + i, layer_in.coords)
            box_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                     layer_in.coords, n*lw*lh + i, 0)
            mask_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                      layer_in.coords, n*lw*lh + i, 4)
            scale = 1 if layer_in.background  else tvm_out[obj_index]
            boxes[index] = _get_region_box(tvm_out, layer_in.biases, n, box_index, col,
                                           row, lw, lh, lw*lh)
            if not layer_in.softmax_tree:
                max_element = 0
                for j in range(layer_in.classes):
                    class_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                               layer_in.coords, n*lw*lh + i, layer_in.coords+1+j)
                    prob = scale*tvm_out[class_index]
                    probs[index][j] = prob if prob > thresh else 0
                    max_element = max(max_element, prob)
                probs[index][layer_in.classes] = max_element

    _correct_region_boxes(boxes, lw*lh*layer_in.n, imw, imh, netw, neth, relative)
    return boxes, probs


def do_nms_sort(boxes, probs, total, classes, thresh):
    "Does the sorting based on the threshold values"
    SortableBbox = namedtuple('SortableBbox', ['index_var', 'class_var', 'probs'])

    s = [SortableBbox(0, 0, []) for i in range(total)]
    for i in range(total):
        s[i] = s[i]._replace(index_var=i)
        s[i] = s[i]._replace(class_var=0)
        s[i] = s[i]._replace(probs=probs)

    for k in range(classes):
        for i in range(total):
            s[i] = s[i]._replace(class_var=k)
        s = sorted(s, key=lambda x: x.probs[x.index_var][x.class_var], reverse=True)
        for i in range(total):
            if probs[s[i].index_var][k] == 0:
                continue
            a = boxes[s[i].index_var]
            for j in range(i+1, total):
                b = boxes[s[j].index_var]
                if _box_iou(a, b) > thresh:
                    probs[s[j].index_var][k] = 0
    return boxes, probs

def draw_detections(im, num, thresh, boxes, probs, names, classes):
    "Draw the markings around the detected region"
    for i in range(num):
        labelstr = []
        category = -1
        for j in range(classes):
            if probs[i][j] > thresh:
                if category == -1:
                    category = j
                labelstr.append(names[j])
        if category > -1:
            imc, imh, imw = im.shape
            width = int(imh * 0.006)
            offset = category*123457 % classes
            red = _get_color(2, offset, classes)
            green = _get_color(1, offset, classes)
            blue = _get_color(0, offset, classes)
            rgb = [red, green, blue]
            b = boxes[i]
            left = int((b.x-b.w/2.)*imw)
            right = int((b.x+b.w/2.)*imw)
            top = int((b.y-b.h/2.)*imh)
            bot = int((b.y+b.h/2.)*imh)

            if left < 0:
                left = 0
            if right > imw-1:
                right = imw-1
            if top < 0:
                top = 0
            if bot > imh-1:
                bot = imh-1
            _draw_box_width(im, left, top, right, bot, width, red, green, blue)
            label = _get_label(''.join(labelstr), rgb)
            _draw_label(im, top + width, left, label, rgb)

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
                        _set_pixel(im, i+c, j+r, k, val)#rgb[k] * val)

def _get_label(labelstr, rgb):
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    text = labelstr
    colorText = "black"
    testDraw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    font = ImageFont.truetype("arial.ttf", 25)
    width, height = testDraw.textsize(labelstr, font=font)
    img = Image.new('RGB', (width, height), color=(int(rgb[0]*255), int(rgb[1]*255),
                                                   int(rgb[2]*255)))
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, fill=colorText, font=font)
    opencvImage = np.divide(np.asarray(img), 255)
    return opencvImage.transpose(2, 0, 1)

def _get_color(c, x, max_value):
    c = int(c)
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (float(x)/float(max_value)) * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio -= i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
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
        _draw_box(im, x1+i, y1+i, x2-i, y2-i, r, g, b)
