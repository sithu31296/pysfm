from pysfm import *
from pysfm.matchers import RoMa
from pysfm.registration import rigid_points_registration_numpy
from pysfm.monodepth import MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.viz import draw_matches, vis_matches_3d_effect
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh
from pysfm.utils.metrics import compute_reprojection_error
np.set_printoptions(suppress=True)



def main(im1_path, im2_path):
    num_kpts = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im1, _ = read_image(im1_path)
    im2, _ = read_image(im2_path)
    H, W, _ = im1.shape

    # find 2D-2D correspondences
    matcher = RoMa(device=device, num_kpts=num_kpts)
    results = matcher(im1_path, im2_path)
    kpts1, kpts2 = results['kpts1'], results['kpts2']

    _, mask = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 0.5)
    mask = mask.ravel() == 1

    depth_model = MoGe(device)
    depth1, est_focal1 = depth_model(im1_path)
    depth2, est_focal2 = depth_model(im2_path)
    pts3d1 = backprojection(depth1, est_focal1)
    pts3d2 = backprojection(depth2, est_focal2)

    pts3d1 = pts3d1[kpts1[mask, 1], kpts1[mask, 0], :]
    pts3d2 = pts3d2[kpts2[mask, 1], kpts2[mask, 0], :]

    K = np.array([
        [est_focal1, 0, W/2],
        [0, est_focal1, H/2],
        [0, 0, 1]
    ], dtype=np.float32)

    # # F, inlier_mask = solver(kpts1, kpts2)
    # R, t, n_inliers = solver(kpts1, kpts2, K, K, depth1, depth2)
    s = 1
    R, t = rigid_points_registration_numpy(pts3d2, pts3d1, compute_scaling=False)

    pts3d1 = backprojection(depth1, est_focal1).reshape(-1, 3)
    pts3d2 = backprojection(depth2, est_focal2).reshape(-1, 3)

    pts3d2 = (s * (R @ pts3d2.T) + t[:, None]).T

    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[:3, :3] = R
    second_pose[:3, -1] = t

    visualize_pcd_with_cams_trimesh(
        [pts3d1, pts3d2], 
        [est_focal1, est_focal2],
        [first_pose, second_pose],
        [im1, im2],
        masks=None,
        point_size=3,
        cam_size=0.2
    )


if __name__ == '__main__':
    im1 = "assets/statue/img0.jpg"
    im2 = "assets/statue/img1.jpg"

    main(im1, im2)