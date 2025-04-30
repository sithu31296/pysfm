from pysfm import *
from pysfm.matchers import RoMa
from pysfm.registration.deformable import DeformableRegistration
from pysfm.monodepth import MoGe
from pysfm.utils.imageio import read_image
from pysfm.utils.geometry import backprojection
from pysfm.utils.viz import visualize_pcd_with_cams_trimesh, visualize_pcd_trimesh
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

    K = np.array([
        [est_focal1, 0, W/2],
        [0, est_focal1, H/2],
        [0, 0, 1]
    ], dtype=np.float32)

    reg = DeformableRegistration(pts3d2, pts3d1)
    reg.register()

    pts3d1 = backprojection(depth1, est_focal1).reshape(-1, 3)
    pts3d2 = backprojection(depth2, est_focal2).reshape(-1, 3)
    pts3d2 += reg.get_registration_parameters(pts3d2)

    visualize_pcd_trimesh([pts3d1, pts3d2], [im1, im2])


if __name__ == '__main__':
    im1 = "assets/statue/img0.jpg"
    im2 = "assets/statue/img1.jpg"

    main(im1, im2)