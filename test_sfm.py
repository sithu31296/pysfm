from pysfm import *
from pysfm.utils.imageio import load_images
from pysfm.matchers.sift import SIFT
from pysfm.matchers.flann import FLANN
from pysfm.utils.geometry import *
from pysfm.solvers.essential_matrix import compute_E, compute_P_from_E
from pysfm.structure.triangulation import triangulate_points
from pysfm.utils.viz import draw_matches
from scipy.optimize import least_squares




def maskout_points(points, mask):
    return points[mask.ravel() > 0]


def sequential_matching(matcher, features):
    """Match all features (Sequential matching)
    Update: for now, lets assume all images can be matched
    Args:
        keypoints       (n_imgs, n_points, 2)
        features        (n_imgs, n_points, 128)
    Returns:
        list of matches with shape [(n_matched_points, 4), ...]    first 2 -> image1, last 2 -> image2
        list of matches = n_imgs - 1
    """
    all_matches = []
    for i in range(len(features)-1):
        matches = matcher.match(features[i], features[i+1])
        all_matches.append(matches)
    return all_matches


def get_matched_points(p1, p2, matches):
    src_pts = np.array([p1[m] for m, n in matches])
    dst_pts = np.array([p2[n] for m, n in matches])
    return src_pts, dst_pts


def reconstruct_structure(K, RT1, RT2, px1, px2):
    P1, P2 = K @ RT1, K @ RT2
    structure = triangulate_points(P1, P2, px1, px2)
    return structure


def initialize_structure(all_matches, all_keypoints, K: np.ndarray):
    """Initialization of the structure from chosen two images
    Args:
        K:          (3, 3)
        keypoints:  (n_imgs, n_points, 2)   n_points = 1024
        colors:     (n_imgs, n_points, 3)
        features    (m_imgs, n_points, 128)
        images      (n_imgs)
    """
    px1, px2 = get_matched_points(all_keypoints[0], all_keypoints[1], all_matches[0])
    E, mask = compute_E(K, px1, px2)
    R2, T2, mask = compute_P_from_E(E, px1, px2, K, mask)
    px1, px2 = maskout_points(px1, mask), maskout_points(px2, mask) # (n_f_points, 2)

    RT1 = np.zeros((3, 4))
    RT1[:3, :3] = np.eye(3)
    RT1[:3, -1] = np.zeros((3, 1)).T

    RT2 = np.zeros((3, 4))
    RT2[:3, :3] = R2
    RT2[:3, -1] = T2.T

    structure = reconstruct_structure(K, RT1, RT2, px1, px2)

    correspond_struct_idx = []
    for kpts in all_keypoints:
        correspond_struct_idx.append(np.ones(len(kpts)) * -1)
    correspond_struct_idx = np.array(correspond_struct_idx).astype(int) # (n_imgs, 1024)
    
    idx = 0
    for i, (m, n) in enumerate(all_matches[0]):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][int(m)] = idx
        correspond_struct_idx[1][int(n)] = idx
        idx += 1
    return structure, correspond_struct_idx, RT1, RT2



def get_2D3D_correspondences(matches, structure, structure_indices, keypoints):
    obj_points, img_points = [], []
    for m, n in matches:
        struct_idx = structure_indices[m]
        if struct_idx < 0:
            continue
        obj_points.append(structure[struct_idx])
        img_points.append(keypoints[n])
    return np.array(obj_points), np.array(img_points)


def fusion_structure(matches, structure, next_structure, struct_indices, next_struct_indices):
    for i, (m, n) in enumerate(matches):
        struct_idx = struct_indices[m]
        if struct_idx >= 0:
            next_struct_indices[n] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis=0)
        struct_indices[m] = next_struct_indices[n] = len(structure) - 1
    return struct_indices, next_struct_indices, structure



def incremental_reconstruction(index, all_keypoints, all_matches, structure, all_structure_indices, K, RTs):
    obj_points, img_points = get_2D3D_correspondences(all_matches[index], structure, all_structure_indices[i], all_keypoints[i+1])
    _, R, T, _ = cv2.solvePnPRansac(obj_points, img_points, K, None)
    R, _ = cv2.Rodrigues(R)
    RT = np.zeros((3, 4))
    RT[:3, :3] = R
    RT[:3, -1] = T.T
    px1, px2 = get_matched_points(all_keypoints[index], all_keypoints[index+1], all_matches[index])
    next_structure = reconstruct_structure(K, RTs[i], RT, px1, px2)
    all_structure_indices[i], all_structure_indices[i+1], structure = fusion_structure(all_matches[i], structure, next_structure, all_structure_indices[i], all_structure_indices[i+1])
    return structure, all_structure_indices, RT


def bundle_adjustment(projections, K, structure, structure_indices, all_keypoints):
    for i in range(len(all_keypoints)):
        p3D_ids = structure_indices[i]
        keypoints = all_keypoints[i]
        RT = projections[i]
        R = RT[:3, :3]
        T = RT[:3, -1]
        r, _ = cv2.Rodrigues(R)

        for j in range(len(p3D_ids)):
            p3D_id = int(p3D_ids[j])
            if p3D_id < 0:
                continue
            
            p3d = structure[p3D_id]
            p2d = keypoints[j]
            reproj_pix, J = cv2.projectPoints(p3d.reshape(1, 1, 3), r, T, K, None)
            reproj_pix = reproj_pix.reshape(2)
            error = abs(p2d - reproj_pix)
            print(error)
    return structure


if __name__ == '__main__':
    image_folder = "assets/ship"
    images, image_paths = load_images(image_folder)
    K = np.array([
        [600, 0, 400],
        [0, 600, 400],
        [0, 0, 1]
    ]).astype(np.float32)
    descriptor = SIFT()
    matcher = FLANN()
    all_keypoints, all_colors, all_features, images, image_paths = descriptor(images, image_paths)
    all_matches = sequential_matching(matcher, all_features)
    structure, structure_indices, RT0, RT1 = initialize_structure(all_matches, all_keypoints, K)
    RTs = [RT0, RT1]

    for i in range(1, len(all_matches)):
        structure, structure_indices, RT = incremental_reconstruction(i, all_keypoints, all_matches, structure, structure_indices, K, RTs)
        RTs.append(RT)

    structure = bundle_adjustment(RTs, K, structure, structure_indices, all_keypoints)
    print(structure.shape)

    # fig = plt.figure()
    # fig.suptitle('3D reconstructed', fontsize=16)
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2])
    # ax.set_xlabel('x axis')
    # ax.set_ylabel('y axis')
    # ax.set_zlabel('z axis')
    # # ax.view_init(elev=135, azim=90)
    # plt.show()
