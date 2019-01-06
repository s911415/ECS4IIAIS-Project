from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf

TF_RCNN_PATH = "T:/lab6/tf-faster-rcnn-windows"
sys.path.append(os.path.join(TF_RCNN_PATH, ""))
sys.path.append(os.path.join(TF_RCNN_PATH, "lib"))
sys.path.append(os.path.join(TF_RCNN_PATH, "tools"))

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from nets.resnet_v1 import resnetv1

iter_cnt = sys.argv[2]

MODEL_PATH = os.path.join(TF_RCNN_PATH,
                          "output/res101/voc_2019_train/default/res101_faster_rcnn_iter_{}.ckpt".format(iter_cnt))

CLASSES = ('__background__',
           '1', '5', '10', '50')

# from train_faster_rcnn
ANCHOR_SCALES = [8, 16, 32]
ANCHOR_RATIOS = [0.5, 1, 2]
CONF_THRESH = 0.8
NMS_THRESH = 0.3

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.USE_GPU_NMS = False

# model path
tfmodel = MODEL_PATH

print(tfmodel)
if not os.path.isfile(tfmodel + '.meta'):
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta'))

# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = False
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.75

# init session
sess = tf.Session(config=tfconfig)
# load network
net = resnetv1(num_layers=101)
net.create_architecture("TEST", len(CLASSES), tag='default',
                        anchor_scales=cfg.ANCHOR_SCALES,
                        anchor_ratios=cfg.ANCHOR_RATIOS
                        )
saver = tf.train.Saver()
saver.restore(sess, tfmodel)

print('Loaded network {:s}'.format(tfmodel))

ANNOTATIONS_TEMPLATE_CONTENT = ''
ANNOTATIONS_TEMPLATE_OBJECT_CONTENT = """<object>
        <name></name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin></xmin>
            <ymin></ymin>
            <xmax></xmax>
            <ymax></ymax>
        </bndbox>
    </object>"""


def predict_and_write(im_path):
    global sess, ANNOTATIONS_TEMPLATE_CONTENT, ANNOTATIONS_TEMPLATE_OBJECT_CONTENT
    dir_path = os.path.dirname(im_path)
    filename, ext = os.path.splitext(os.path.basename(im_path))
    im = cv2.imread(im_path)
    root = ET.fromstring(ANNOTATIONS_TEMPLATE_CONTENT)
    root.find('./size/height').text = str(im.shape[0])
    root.find('./size/width').text = str(im.shape[1])
    root.find('./size/depth').text = str(im.shape[2])
    root.find('./filename').text = str(filename + ext)
    root.find('./folder').text = str("images")
    scores, boxes = im_detect(sess, net, im)

    for cls_ind in range(1, len(CLASSES)):
        class_name = CLASSES[cls_ind]
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        thresh = CONF_THRESH

        ###
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) > 0:
            # im = im[:, :, (2, 1, 0)]
            for i in inds:
                bbox = dets[i, :4]
                obj = ET.fromstring(ANNOTATIONS_TEMPLATE_OBJECT_CONTENT)
                obj.find('./name').text = class_name
                keys = "xmin,ymin,xmax,ymax".split(",")
                for j in range(4):
                    obj.find('./bndbox/' + keys[j]).text = str(int(bbox[j]))
                root.append(obj)

    tree1 = ET.ElementTree(root)
    xml_path = os.path.join(dir_path, filename + ".xml")
    tree1.write(xml_path, encoding='utf8')


def load_template():
    global ANNOTATIONS_TEMPLATE_CONTENT
    with open("annotations_template.xml", "r") as f:
        ANNOTATIONS_TEMPLATE_CONTENT = f.read()


if __name__ == '__main__':
    load_template()
    img_files = glob.glob(os.path.join(sys.argv[1], "*.jpg"))
    for path in img_files:
        print("Predicting", path)
        predict_and_write(path)
