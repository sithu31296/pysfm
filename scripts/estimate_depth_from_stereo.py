from pysfm import *
from pysfm.monodepth import MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh
from pysfm.utils.transforms import focal_mm_to_pixels
from pysfm.segmenters.sky import SkyRemover
np.set_printoptions(suppress=True)


def main(root):
    root = Path(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    left_image = cv2.imread(root / "view1.png")
    right_image = cv2.imread(root / "view5.png")

    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    H, W, _ = left_image.shape

    focal = 3740
    baseline = 0.16 # in meters

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=225,  
        blockSize=15,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    print(disparity.min(), disparity.max())

    gt_disparity = cv2.imread(root / "disp1.png", cv2.IMREAD_UNCHANGED) 
    print(gt_disparity.min(), gt_disparity.max())

    depth = np.where(disparity > 0, (baseline * focal) / disparity, 0)
    gt_depth = np.where(gt_disparity > 0, (baseline * focal) / gt_disparity, 0)

    print(f"Original Image")
    print(f"\tImage Size (H x W)\t: {left_image.shape[0]} x {left_image.shape[1]}")
    print(f"\tFocal Length in Pixels\t: {focal:.2f}")
    print(f"\tDepth Ranges in Meters\t: {depth.min():.2f} - {depth.max():.2f}")
    print(f"\tGT Depth Ranges in Meters\t: {gt_depth.min():.2f} - {gt_depth.max():.2f}")

    # pts3d = backprojection(depth, focal)
    # visualize_pcd_with_cams_trimesh([pts3d], [focal], [np.eye(4)], [left_image], None, point_size=3, cam_size=0.2)


if __name__ == '__main__':
    root = "./assets/stereo"
    main(root)