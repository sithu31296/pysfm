from pysfm import *
sys.path.append(str(Path(__file__).parent / "../../third_party"))
from LoFTR.src.loftr import LoFTR as LoFTRBase, default_cfg
from pysfm.utils.imageio import rgb2gray


class LoFTR:
    def __init__(self, weights_path, device=None, outdoor=True) -> None:
        self.ransac_threshold = 10
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if weights_path is None:
            weights_path = "third_party/LoFTR/weights/outdoor_ot.ckpt" if outdoor else "pysfm/third_party/LoFTR/weights/indoor_ot.ckpt"

        self.matcher = LoFTRBase(default_cfg)
        self.matcher.load_state_dict(torch.load(weights_path)['state_dict'], strict=False)
        self.matcher = self.matcher.eval().to(self.device)
    
    def transform_to_tensor(self, image):
        return torch.from_numpy(image / 255.).float()[None, None].to(self.device)
    
    def __call__(self, image1, image2):
        gray1, gray2 = rgb2gray(image1), rgb2gray(image2)
        gray1, gray2 = self.transform_to_tensor(gray1), self.transform_to_tensor(gray2)

        # LoFTR needs resolution multiple of 8. 
        # if not, we pad 0's to get to a multiple of 8
        if gray1.shape[3] % 8 != 0:
            pad_bottom = gray1.shape[2] % 8
            pad_right = gray1.shape[3] % 8
            pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
            gray1 = pad_fn(gray1)

        if gray2.shape[3] % 8 != 0:
            pad_bottom = gray2.shape[2] % 8
            pad_right = gray2.shape[3] % 8
            pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
            gray2 = pad_fn(gray2)

        with torch.no_grad():
            batch = {"image0": gray1, "image1": gray2}
            self.matcher(batch)
            pts1 = batch['mkpts0_f'].cpu().numpy()
            pts2 = batch['mkpts1_f'].cpu().numpy()

        # # constrain matches to fit homography
        # retval, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
        # mask = mask.ravel() # len(pts1)

        # # select inlier points
        # pts1, pts2 = pts1[mask == 1], pts2[mask == 1]
        return pts1, pts2
    

