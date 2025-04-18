from pysfm import *
from pysfm.monodepth import DepthPro, MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh
from pysfm.segmenters.sky import SkyRemover
np.set_printoptions(suppress=True)


def main(root, resize=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # depth_model = DepthPro(device)
    depth_model = MoGe(device)
    sky_remover = SkyRemover()
    image = read_image(root)[0]
    sky_mask = sky_remover(root)
    depth, est_focal = depth_model(root)
    print(f"Original Image")
    print(f"\tImage Size (H x W)\t: {image.shape[0]} x {image.shape[1]}")
    print(f"\tFocal Length in Pixels\t: {est_focal:.2f}")
    print(f"\tDepth Ranges in Meters\t: {depth.min():.2f} - {depth.max():.2f}")

    pts3d = backprojection(depth, est_focal)
    mask = sky_mask
    visualize_pcd_with_cams_trimesh([pts3d], [est_focal], [np.eye(4)], [image], [mask], point_size=3, cam_size=0.2)


if __name__ == '__main__':
    root = "./assets/statue/img0.jpg"
    # root = "./assets/south-building/images/P1180141.JPG"
    # resize = (1920, 1440)
    resize = None
    main(root, resize)