from pysfm import *
from moge.model.v1 import MoGeModel
from pysfm.utils.imageio import read_images
from pysfm.utils.transforms import focal_to_fov


class MoGe(nn.Module):
    """
    Note: MoGe estimates scale-invariant point and depth maps
    """
    def __init__(self, device="cuda"):
        super().__init__()
        # load the model from huggingface hub
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        self.device = device
        self.max_depth = np.inf
    
    def forward(self, path: Union[str, list, Path], focals: List[float] = None):
        images, gt_focals = read_images(path)
        depths, est_focals = [], []
        
        # if some images don't have metadata and want to use provided focals in depth estimation model
        if focals is not None:
            if isinstance(focals, int):
                focals = [focals] * len(images)
            for i in range(len(gt_focals)):
                if gt_focals[i] is None:
                    gt_focals[i] = focals[i]

        for image, focal in zip(images, gt_focals):
            image = torch.tensor(image / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            if focal is not None:
                fov = np.degrees(focal_to_fov(focal, image.shape[2]))
            else:
                fov = None
            output = self.model.infer(
                image, 
                fov, 
                resolution_level=9,     # higher value means more tokens and finer details but slower
                apply_mask=True,
                force_projection=True,
                use_fp16=True
            )
            points = output['points'].detach().cpu().numpy().astype(np.float32)
            depth = output['depth'].detach().cpu().numpy().astype(np.float32)
            depth[depth >= np.inf] = 0.0
            intrinsics = output['intrinsics'].detach().cpu().numpy().astype(np.float32)
            focal = intrinsics[0, 0] * image.shape[2]
            depths.append(depth)
            est_focals.append(focal)

        if not isinstance(path, list):
            if Path(path).is_file():
                return depths[0], est_focals[0]
        return depths, est_focals
    
