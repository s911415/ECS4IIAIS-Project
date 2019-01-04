import glob
import os
import secrets
import sys
from pathlib import Path
from shutil import copyfile

secrets_generator = secrets.SystemRandom()

if __name__ == '__main__':
    src_dir = sys.argv[1]
    dst_dir = sys.argv[2]
    img_dirs = ['Annotations', 'JPEGImages', os.path.join('ImageSets', 'Main')]
    for dir_path in img_dirs:
        d = Path(os.path.join(dst_dir, dir_path))
        d.mkdir(parents=True, exist_ok=True)

    labeled_filename = []
    img_fn_map = {}
    xml_files = glob.glob(os.path.join(src_dir, "*.xml"))
    for xml_file in xml_files:
        with open(xml_file, 'r', encoding='utf8') as f:
            txt = f.read()
            if "<name>Unknown</name>" not in txt:
                start_txt = "<filename>"
                s = txt.index(start_txt) + len(start_txt)
                e = txt.index("</filename>", s)
                img_fn = txt[s:e]

                img_path = os.path.join(src_dir, img_fn)
                if os.path.exists(img_path):
                    # if not img_fn.lower().endswith(".jpg"):
                    #    print("Not jpeg file: " + img_path, file=sys.stderr)
                    #    continue
                    xml_fn = os.path.basename(xml_file)
                    labeled_filename.append(xml_fn)
                    img_fn_map[xml_fn] = img_fn

    train_f = open(os.path.join(dst_dir, 'ImageSets', 'Main', 'train.txt'), 'w', encoding='utf8')
    test_f = open(os.path.join(dst_dir, 'ImageSets', 'Main', 'test.txt'), 'w', encoding='utf8')

    for fn in labeled_filename:
        xml_path = os.path.join(src_dir, fn)
        img_path = os.path.join(src_dir, img_fn_map[fn])

        copyfile(xml_path, os.path.join(dst_dir, 'Annotations', fn))
        copyfile(img_path, os.path.join(dst_dir, 'JPEGImages', img_fn_map[fn]))

        p = secrets_generator.randint(0, 100)
        fn_wo_ext = os.path.splitext(fn)[0]
        fio = train_f
        if p > 70:
            fio = test_f

        fio.write(fn_wo_ext)
        fio.write("\n")

    train_f.close()
    test_f.close()
