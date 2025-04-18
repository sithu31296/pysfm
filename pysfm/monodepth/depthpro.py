from pysfm import *
import depth_pro
from pysfm.utils.imageio import read_images


class DepthPro(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model, self.transform = depth_pro.create_model_and_transforms(device=device)
        self.model.eval()

        self.max_depth = 200
    
    def forward(self, path: Union[str, list, Path], focals: List[float] = None):
        images, gt_focals = read_images(path)
        depths, est_focals = [], []
        
        # if some images don't have metadata and want to use provided focals in depth estimation model
        if focals is not None:
            for i in range(len(gt_focals)):
                if gt_focals[i] is None:
                    gt_focals[i] = focals[i]

        for image, focal in zip(images, gt_focals):
            image = self.transform(image)
            pred = self.model.infer(image, f_px=focal)
            depth = pred['depth'].detach().cpu().numpy().astype(np.float32)   # depth in meters
            depth[depth >= self.max_depth] = 0.0
            if focal is None:
                focal = pred['focallength_px'].detach().cpu().numpy().astype(np.float32)   # focal length in pixels
            depths.append(depth)
            est_focals.append(focal)

        if not isinstance(path, list):
            if Path(path).is_file():
                return depths[0], est_focals[0]
        return depths, est_focals
    
