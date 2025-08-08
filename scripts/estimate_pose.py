from pysfm import *
from pysfm.matchers import RoMa
from pysfm.solvers import *
from pysfm.monodepth import MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh
from pysfm.utils.metrics import compute_reprojection_error
np.set_printoptions(suppress=True)



def main(im1_path, im2_path):
    mode = "relative"
    num_kpts = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    im1, _ = read_image(im1_path)
    im2, _ = read_image(im2_path)

    matcher = RoMa(device=device, num_kpts=num_kpts)
    depth_model = MoGe(device)

    results = matcher(im1_path, im2_path)
    kpts1, kpts2 = results['kpts1'], results['kpts2']
    
    H, W, _ = im1.shape
    depth1, est_focal1 = depth_model(im1_path)
    depth2, est_focal2 = depth_model(im2_path)
    pts3d1 = backprojection(depth1, est_focal1).reshape(-1, 3)
    pts3d2 = backprojection(depth2, est_focal2)

    K = np.array([
        [est_focal1, 0, W/2],
        [0, est_focal1, H/2],
        [0, 0, 1]
    ], dtype=np.float32)

    if mode == 'relative':
        # solver = HomographySolver(method="opencv")
        # solver = FundamentalSolver(method="poselib")
        solver = EssentialSolver(method="poselib")
        R, t, n_inliers, inliers_mask = solver(kpts2, kpts1, K, K)

        scale_solver = RelativePoseScaleSolver(ransac_scale_threshold=0.05, use_ransac=True)
        real_scale = scale_solver(kpts2, kpts1, depth2, depth1, K, K, R, t, inliers_mask)
        t *= real_scale
    else:
        solver = PnPSolver(reproj_threshold=12.0, method="poselib")
        points3d = pts3d2[kpts2[:, 1], kpts2[:, 0], :].astype(np.float32)
        R, t, n_inliers, inliers_mask = solver(kpts1, points3d, K)
    
    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[:3, :3] = R
    second_pose[:3, -1:] = t

    print(f"\tNum. of Matched Correspondences\t: {kpts1.shape[0]}")
    print(f"\tNum. of Inlier Correspondences\t: {n_inliers}")
    print(f"\tOriginal Image Resolution\t: {im1.shape[:2]} and {im2.shape[:2]} [H, W]")
    print(f"\tImage Resolution used in Model\t: {matcher.model_res}")
    print(f"\tFocal Length in Pixels\t\t: {est_focal1:.2f}")

    pts3d2 = ((R @ pts3d2.reshape(-1, 3).T) + t).T
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