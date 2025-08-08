from pysfm import *
from pysfm.matchers import RoMa
from pysfm.registration import rigid_points_registration_numpy
from pysfm.monodepth import MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh
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

    depth_model = MoGe(device)
    depth1, est_focal1 = depth_model(im1_path)
    depth2, est_focal2 = depth_model(im2_path)
    pts3d1 = backprojection(depth1, est_focal1)
    pts3d2 = backprojection(depth2, est_focal2)

    pts3d1 = pts3d1[kpts1[:, 1], kpts1[:, 0], :]
    pts3d2 = pts3d2[kpts2[:, 1], kpts2[:, 0], :]

    R, t, s = rigid_points_registration_numpy(pts3d2, pts3d1, compute_scaling=True)

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