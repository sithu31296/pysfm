from pysfm import *
from pysfm.matchers import RoMa, LoFTR
from pysfm.utils.imageio import read_image
from pysfm.utils.viz import draw_matches, vis_matches_3d_effect


def main(im1_path, im2_path):
    sample = 100
    num_kpts = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im1, _ = read_image(im1_path)
    im2, _ = read_image(im2_path)

    matcher = RoMa(device=device, num_kpts=num_kpts)
    results = matcher(im1_path, im2_path)
    kpts1, kpts2 = results['kpts1'], results['kpts2']
    print(f"\tNum. of Matched Keypoints\t: {kpts1.shape[0]}")
    print(f"\tOriginal Image Resolution\t: {im1.shape[:2]} and {im2.shape[:2]} [H, W]")
    print(f"\tImage Resolution used in Model\t: {matcher.model_res}")

    draw_matches(kpts1[:sample, :], kpts2[:sample, :], im1, im2)
    if "dense_matches" in results.keys():
        vis_matches_3d_effect(im1, im2, results['dense_matches'], matcher.model_res, results['certainty'])



if __name__ == '__main__':
    im1 = "assets/statue/img0.jpg"
    im2 = "assets/statue/img1.jpg"

    main(im1, im2)