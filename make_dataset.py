import glob
import multiprocessing.dummy as mp
import os
import secrets
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import cv2
import numpy as np

NEW_DATA_DIR = './out/new_dataset/'

secrets_generator = secrets.SystemRandom()
START_TIME = int(time.time())


def extract_object(e: Element):
    return e.find('./name').text, \
           int(e.find('./bndbox/xmin').text), \
           int(e.find('./bndbox/ymin').text), \
           int(e.find('./bndbox/xmax').text), \
           int(e.find('./bndbox/ymax').text)


def change_hls(im, value):
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    offset_h, offset_l, offset_s = value

    def change_val(ch, val, lower, upper):
        if val == 0:
            return

        if val > 0:
            lim = upper - val
            ch[ch >= lim] = upper
            ch[ch < lim] += val
        else:
            lim = lower - val
            ch[ch <= lim] = lower
            ch[ch > lim] -= -val

    change_val(h, offset_h, 0, 180)
    change_val(l, offset_l, 0, 255)
    change_val(s, offset_s, 0, 255)

    final_hsv = cv2.merge((h, l, s))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HLS2BGR)
    return img


def generate_new_image(xml_path: str, repeat_cnt=None):
    if repeat_cnt is None:
        repeat_cnt = secrets_generator.randint(1, 3)

    xml_doc = ET.parse(xml_path)
    path_elem = xml_doc.find('./path')
    if path_elem is not None:
        xml_doc.getroot().remove(path_elem)

    file_name = xml_doc.find('./filename').text
    objects = xml_doc.findall('./object')
    objs = list(map(extract_object, objects))
    labeled_cnt = len(list(filter(lambda e: e[0] != 'Unknown', objs)))

    if labeled_cnt == len(objs):
        im_path = os.path.join(os.path.dirname(xml_path), file_name)
        img_filename, img_file_extension = os.path.splitext(os.path.basename(im_path))
        im = cv2.imread(im_path)
        if im is None:
            print("Cannot open " + im_path, file=sys.stderr)
            return

        for cnt in range(repeat_cnt):
            h, w, c = im.shape
            mask = np.zeros(im.shape, dtype="uint8")
            for label, left, top, right, bottom in objs:
                center = (
                    int((left + right) / 2),
                    int((top + bottom) / 2)
                )
                r = int(max(right - left, bottom - top) / 2)
                cv2.circle(mask, center, r, (255, 255, 255), -1)

            offset_h, offset_l, offset_s = \
                secrets_generator.randint(-10, 10), \
                secrets_generator.randint(-50, 50), \
                secrets_generator.randint(-10, 10)

            # blur_k = secrets_generator.randint(1, 4)

            bg_mask = cv2.bitwise_not(mask)
            coin_im = cv2.bitwise_and(im, mask)
            # coin_im = cv2.blur(coin_im, (blur_k, blur_k))
            bg_im = im.copy()

            # change h on background
            bg_im = change_hls(bg_im, (offset_h, 0, 0))
            bg_im = cv2.bitwise_and(bg_im, bg_mask)

            combined_im = bg_im.copy()
            cv2.addWeighted(coin_im, 1, combined_im, 1, 0, combined_im)
            combined_im = change_hls(combined_im, (0, offset_l, offset_s))

            new_filename = os.path.join(NEW_DATA_DIR, str(START_TIME) + '_' + img_filename) + '_' + str(uuid.uuid4())
            new_img_name = new_filename + img_file_extension
            new_xml_name = new_filename + '.xml'

            cv2.imwrite(new_img_name, combined_im)
            xml_doc.find('./filename').text = os.path.basename(new_img_name)
            xml_doc.write(new_xml_name, encoding='utf8')
            # cv2.imshow("coin", coin_im)
            # cv2.imshow("bg", bg_im)
            # cv2.imshow(str(uuid.uuid4()), combined_im)


def generate_new_image_with_blur(xml_path: str, repeat_cnt=None):
    if repeat_cnt is None:
        repeat_cnt = secrets_generator.randint(1, 3)

    xml_doc = ET.parse(xml_path)
    path_elem = xml_doc.find('./path')
    if path_elem is not None:
        xml_doc.getroot().remove(path_elem)

    file_name = xml_doc.find('./filename').text
    objects = xml_doc.findall('./object')
    objs = list(map(extract_object, objects))
    labeled_cnt = len(list(filter(lambda e: e[0] != 'Unknown', objs)))

    if labeled_cnt == len(objs):
        im_path = os.path.join(os.path.dirname(xml_path), file_name)
        img_filename, img_file_extension = os.path.splitext(os.path.basename(im_path))
        im = cv2.imread(im_path)
        if im is None:
            print("Cannot open " + im_path, file=sys.stderr)
            return

        for cnt in range(repeat_cnt):
            h, w, c = im.shape
            blur_k = secrets_generator.randint(2, 6)
            bg_im = im.copy()
            combined_im = cv2.blur(bg_im, (blur_k, blur_k))

            new_filename = os.path.join(NEW_DATA_DIR, str(START_TIME) + '_' + img_filename) + '_' + str(uuid.uuid4())
            new_img_name = new_filename + img_file_extension
            new_xml_name = new_filename + '.xml'

            cv2.imwrite(new_img_name, combined_im)
            xml_doc.find('./filename').text = os.path.basename(new_img_name)
            xml_doc.write(new_xml_name, encoding='utf8')
            # cv2.imshow("coin", coin_im)
            # cv2.imshow("bg", bg_im)
            # cv2.imshow(str(uuid.uuid4()), combined_im)


if __name__ == '__main__':
    # for i in range(1):
    #     generate_new_image('T:\emb_fin\images\\190103_005.xml', 2)
    # while True:
    #     cv2.waitKey(500)
    if not os.path.exists(NEW_DATA_DIR):
        os.mkdir(NEW_DATA_DIR)

    xml_files = glob.glob(os.path.join(sys.argv[1], "*.xml"))
    p = mp.Pool()
    p.map(generate_new_image_with_blur, xml_files)
    p.close()
    p.join()
    # for xml_path in xml_files:
    #     try:
    #         generate_new_image(xml_path)
    #     except Exception as e:
    #         print("Error when processing image: " + xml_path)
    #         raise e
    # pass
