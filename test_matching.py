from pysfm import *
from pysfm.utils.imageio import load_images
from pysfm.matchers.loftr import LoFTR
from pysfm.utils.viz import draw_matches

image_folder = "assets/test31"
images, image_paths = load_images(image_folder, (1280, 720))
matcher = LoFTR(None)


def rescale_points(orig_img_size, new_img_size, points):
    # print(points[:, 0].min(), points[:, 0].max())
    # print(points[:, 1].min(), points[:, 1].max())
    w0, h0 = orig_img_size
    w1, h1 = new_img_size
    sh, sw = h0/h1, w0/w1
    points[:, 0] /= sw
    points[:, 1] /= sh
    return points


for i in range(len(images)-1):
    image1, image2 = images[i], images[i+1]
    pts1, pts2 = matcher(image1, image2)
    pts1 = rescale_points((1280, 720), (512, 288), pts1)
    pts2 = rescale_points((1280, 720), (512, 384), pts2)
    pts1, pts2 = pts1.astype(int), pts2.astype(int)
    image1 = cv2.resize(image1, (512, 288))
    image2 = cv2.resize(image2, (512, 384))
    draw_matches(pts1, pts2, image1, image2)
    break