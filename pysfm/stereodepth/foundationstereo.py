from pysfm import *
from pysfm.utils.imageio import read_image
from omegaconf import OmegaConf
sys.path.append(str(Path(__file__).parent / "../../third_party/FoundationStereo"))
from core.foundation_stereo import FoundationStereo as FoundationStereoModel


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
          max_side = max(self.ht, self.wd)
          pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
          pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
          pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
          pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    

class FoundationStereo(nn.Module):
    """
    Note: MoGe estimates scale-invariant point and depth maps
    """
    def __init__(self, device="cuda"):
        super().__init__()
        checkpoint_dir = Path("checkpoints")
        cfg = OmegaConf.load(checkpoint_dir / "23-51-11" / "cfg.yaml")
        if "vit_size" not in cfg:
            cfg['vit_size'] = "vitl"
        args = OmegaConf.create(cfg)

        z_far = 10  # max depth to clip in point cloud
        self.valid_iters = 32    # number of flow-field updates during forward pass
        remove_invisible = 1    # remove non-overlapping observations between left and right image
        denoise_cloud = 1       # whether to denoise the point cloud
        denoise_nb_points = 30  # number of points to consider for radius outlier removal
        denoise_radius = 0.03   # radius to use for outlier removal

        self.model = FoundationStereoModel(args)
        ckpt = torch.load(str(checkpoint_dir / "23-51-11" / "model_best_bp2.pth"), map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()
        self.device = device
        self.max_depth = np.inf
    
    def forward(self, left_path: Union[str, list, Path], right_path: Union[str, list, Path], focal: float = None):
        left_image, focal = read_image(left_path)
        right_image, focal = read_image(right_path)
        H, W, _ = left_image.shape

        left_image = torch.as_tensor(left_image).to(self.device).float()[None].permute(0, 3, 1, 2)
        right_image = torch.as_tensor(right_image).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(left_image.shape, divis_by=32, force_square=False)
        left_image, right_image = padder.pad(left_image, right_image)

        with torch.no_grad():
            disp = self.model.forward(left_image, right_image, iters=self.valid_iters, test_mode=True)

        disp = padder.unpad(disp.float())
        disp = disp.detach().cpu().numpy().reshape(H, W)
        # depths, est_focals = [], []
        
        # # if some images don't have metadata and want to use provided focals in depth estimation model
        # if focals is not None:
        #     for i in range(len(gt_focals)):
        #         if gt_focals[i] is None:
        #             gt_focals[i] = focals[i]

        # for image, focal in zip(images, gt_focals):
        #     image = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
        #     if focal is not None:
        #         fov = np.degrees(focal_to_fov(focal, image.shape[2]))
        #     else:
        #         fov = None
        #     output = self.model.infer(
        #         image, 
        #         fov, 
        #         resolution_level=9,     # higher value means more tokens and finer details but slower
        #         apply_mask=True,
        #         force_projection=True,
        #         use_fp16=True
        #     )
        #     points = output['points'].detach().cpu().numpy().astype(np.float32)
        #     depth = output['depth'].detach().cpu().numpy().astype(np.float32)
        #     depth[depth >= np.inf] = 0.0
        #     intrinsics = output['intrinsics'].detach().cpu().numpy().astype(np.float32)
        #     focal = intrinsics[0, 0] * image.shape[2]
        #     depths.append(depth)
        #     est_focals.append(focal)

        # if not isinstance(path, list):
        #     if Path(path).is_file():
        #         return depths[0], est_focals[0]
        # return depths, est_focals

        return disp
    
