# Created by yongxinwang at 2019-04-24 14:44
# Last modified by yongxinwang at 2019-04-24 14:44
import cv2
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing as mp


def process_edges_one(im_path, savedir):
    image = cv2.imread(im_path, 0)
    edges = cv2.Canny(image, 100, 200)
    image_name = os.path.basename(im_path).replace('.jpg', '')
    cv2.imwrite(os.path.join(savedir, image_name + '_edges.jpg'), edges)


def process_edges(root):
    all_images = glob.glob(os.path.join(root, "*", "*.jpg"))
    savedir = root.replace("images", "edges")

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    pool = mp.Pool(processes=8)

    for imp in all_images:
        bird_class = os.path.basename(os.path.dirname(imp))
        print(bird_class)
        if not os.path.exists(os.path.join(savedir, bird_class)):
            os.mkdir(os.path.join(savedir, bird_class))

        pool.apply(process_edges_one, args=(imp, os.path.join(savedir, bird_class)))
        # break

if __name__ == "__main__":
    process_edges("/media/hdd3/tmp/CUB_200_2011/images/")
    # process_edges_one("/media/hdd3/tmp/CUB_200_2011/images/")
