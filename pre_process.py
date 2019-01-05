import glob
import os
import sys
import xml.etree.ElementTree as ET

import multiprocessing.dummy as mp
import cv2
import numpy as np

CIRCLE_IMG_TMP_DIR = './out/circles/'
ANNOTATIONS_DIR = './out/Annotations/'
param_label = None
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


def process(img_path: str):
    global ANNOTATIONS_TEMPLATE_CONTENT, param_label
    file_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # cv2.imshow("raw", img)
    ed_img = cv2.erode(img, kernel, iterations=3)
    ed_img = cv2.dilate(ed_img, kernel, iterations=3)

    gray_img = cv2.cvtColor(ed_img, cv2.COLOR_BGR2GRAY)
    edge_output = cv2.Canny(gray_img, 50, 255)
    circles = cv2.HoughCircles(edge_output, cv2.HOUGH_GRADIENT, 2, 100, maxRadius=150)
    if circles is None:
        return

    circles = np.uint16(np.around(circles))

    cimg = img.copy()

    circle_data_list = []
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        circle_data_list.append({
            "left": i[0] - i[2],
            "right": i[0] + i[2],
            "top": i[1] - i[2],
            "bottom": i[1] + i[2],
        })

    # cv2.imshow('detected circles', cimg)
    # cv2.imwrite(os.path.join(CIRCLE_IMG_TMP_DIR, file_name), cimg)

    # write xml
    root = ET.fromstring(ANNOTATIONS_TEMPLATE_CONTENT)
    root.find('./size/height').text = str(img.shape[0])
    root.find('./size/width').text = str(img.shape[1])
    root.find('./size/depth').text = str(img.shape[2])
    root.find('./filename').text = str(file_name)
    root.find('./folder').text = str("images")

    for circle in circle_data_list:
        obj = ET.fromstring(ANNOTATIONS_TEMPLATE_OBJECT_CONTENT)
        obj.find('./name').text = 'Unknown' if param_label is None else param_label
        obj.find('./bndbox/xmin').text = str(circle["left"])
        obj.find('./bndbox/xmax').text = str(circle["right"])
        obj.find('./bndbox/ymin').text = str(circle["top"])
        obj.find('./bndbox/ymax').text = str(circle["bottom"])
        root.append(obj)

    tree1 = ET.ElementTree(root)
    file_name_only, ext = os.path.splitext(file_name)
    xml_path = os.path.join(ANNOTATIONS_DIR, file_name_only + ".xml")
    tree1.write(xml_path, encoding='utf8')


def load_template():
    global ANNOTATIONS_TEMPLATE_CONTENT
    with open("annotations_template.xml", "r") as f:
        ANNOTATIONS_TEMPLATE_CONTENT = f.read()


if __name__ == '__main__':
    for dir_path in [CIRCLE_IMG_TMP_DIR, ANNOTATIONS_DIR]:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    load_template()
    if len(sys.argv) == 3:
        param_label = sys.argv[2]

    p = mp.Pool()
    img_files = glob.glob(os.path.join(sys.argv[1], "*.jpg"))
    p.map(process, img_files)
    p.close()
    p.join()
