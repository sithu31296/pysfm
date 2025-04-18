from pysfm import *
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation
from pysfm.utils.imageio import tensor_to_image
from pysfm.utils.geometry import compute_epipole
from pysfm.utils.transforms import *


Cam_to_Trimesh = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

OPENGL = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


CAM_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 204, 0), (0, 204, 204),
              (128, 255, 255), (255, 128, 255), (255, 255, 128), (0, 0, 0), (128, 128, 128)]


def vis_image(img: Union[Image.Image, np.ndarray, Tensor]):
    img = tensor_to_image(img)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()



def vis_matches_3d_effect(im1, im2, warp, model_res, certainty):
    H, W = model_res
    x1 = (torch.tensor(np.array(im1)) / 255).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).permute(2, 0, 1)

    im1_transfer_rgb = F.grid_sample(x1[None], torch.tensor(warp[:, W:, :2][None]), mode="bilinear", align_corners=False)[0]
    im2_transfer_rgb = F.grid_sample(x2[None], torch.tensor(warp[:,:W, 2:][None]), mode="bilinear", align_corners=False)[0]
    
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2).numpy()
    white_im = np.ones((H,2*W))
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    vis_image(vis_im)


def draw_matches(pts1, pts2, img0, img1):
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    img0_moved = False
    img1_moved = False
    new_img0 = img0.copy()
    new_img1 = img1.copy()
    offset_w = 0
    offset_h = 0
    
    if w1 > w0 or h1 > h0:
        offset_w = w1 - w0
        new_img0 = np.zeros((h1, w1, 3), dtype=np.uint8)
        new_img0[:h0, offset_w:w0, :] = img0
        img0_moved = True
        
    elif w0 > w1 or h0 > h1:
        new_img1 = np.zeros((h0, w0, 3), dtype=np.uint8)
        new_img1[:h1, :w1, :] = img1
        img1_moved = True 

    img = np.concatenate([new_img0, new_img1], axis=1)
    offset = img.shape[1] / 2
    plt.imshow(img)
    for p1, p2 in zip(pts1, pts2):
        if img0_moved:
            plt.scatter([p1[0]+offset_w, p2[0] + offset], [p1[1]+offset_h, p2[1]])
            plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]])
        elif img1_moved:
            plt.scatter([p1[0], p2[0] + offset + offset_w], [p1[1], p2[1] + offset_h])
            plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]])
        else:
            plt.scatter([p1[0], p2[0] + offset], [p1[1], p2[1]])
            plt.plot([p1[0], p2[0] + offset], [p1[1], p2[1]])
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_epipolar_line(p1, p2, F, show_epipole=False):
    """Plot the epipole and epipolar line F*x=0 in an image given the corresponding points.
    """
    lines = np.dot(F, p2)
    pad = np.ptp(p1, 1) * 0.01
    mins = np.min(p1, 1)
    maxes = np.max(p1, 1)

    # epipolar line parameter and values
    xpts = np.linspace(mins[0] - pad[0], maxes[0] + pad[0], 100)
    for line in lines.T:
        ypts = np.asarray([(line[2] + line[0] * p) / (-line[1]) for p in xpts])
        valid_idx = ((ypts >= mins[1] - pad[1]) & (ypts <= maxes[1] + pad[1]))
        plt.plot(xpts[valid_idx], ypts[valid_idx], linewidth=1)
        plt.plot(p1[0], p1[1], 'ro')

    if show_epipole:
        epipole = compute_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def plot_epipolar_lines(p1, p2, F, show_epipole=False):
    """Plot the points and epipolar lines P1' F P2 = 0
    """
    plt.figure()
    plt.suptitle("Epipolar lines", fontsize=16)

    # plot the epipolar lines on img1 with points p2 from the right side
    # L1 = F * p2
    plt.subplot(1, 2, 1, aspect='equal')
    plot_epipolar_line(p1, p2, F, show_epipole)
    
    # plot the epipolar lines on img2 with points p1 from the left side
    # L2 = F' * p1
    plt.subplot(1, 2, 2, aspect='equal')
    plot_epipolar_line(p2, p1, F.T, show_epipole)


def process_pts_colors(pts, colors=None, masks=None):
    pts = cat_3d(pts)
    if colors is None:
        colors = np.ones_like(pts)
        colors *= np.array([0, 0, 255])
        colors = colors.astype(np.uint8)
    colors = cat_3d(colors)
    if masks is not None:
        masks = cat_1d(masks)
        pts = pts[masks]
        colors = colors[masks]
    return pts, colors


def visualize_pcd_trimesh(pts, imgs=None, masks=None, point_size=3):
    pts, colors = process_pts_colors(pts, imgs, masks)
    scene = trimesh.Scene()
    pcd = trimesh.PointCloud(pts.reshape(-1, 3), colors.reshape(-1, 3))
    scene.add_geometry(pcd, transform=Cam_to_Trimesh)
    scene.show(line_settings={"point_size": point_size})


def visualize_pcd_with_cams_trimesh(pts, focals, c2ws, imgs=None, masks=None, point_size=3, cam_size=0.1):
    pts, colors = process_pts_colors(pts, imgs, masks)
    focals = to_numpy(focals)
    c2ws = to_numpy(c2ws)

    scene = trimesh.Scene()
    pcd = trimesh.PointCloud(pts.reshape(-1, 3), colors.reshape(-1, 3))
    scene.add_geometry(pcd, transform=Cam_to_Trimesh)

    # add each camera
    for i, pose_c2w in enumerate(c2ws):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam_trimesh(scene, pose_c2w, camera_edge_color, imgs[i], focals[i], screen_width=cam_size)

    scene.show(line_settings={"point_size": point_size})




def add_scene_cam_trimesh(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03, marker=None):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=Image.fromarray(image))
        scene.add_geometry(img, transform=Cam_to_Trimesh)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam, transform=Cam_to_Trimesh)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker, transform=Cam_to_Trimesh)
