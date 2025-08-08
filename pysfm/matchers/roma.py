from pysfm import *
from romatch import roma_outdoor, roma_indoor
from pysfm.utils.imageio import read_image


class RoMa:
    def __init__(self, weights_path=None, device="cpu", outdoor=True, num_kpts=10000) -> None:
        self.device = device
        self.num_kpts = num_kpts
        self.model_res = (864, 864)
        if outdoor:
            self.roma_model = roma_outdoor(device, coarse_res=560, upsample_res=864)
        else:
            self.roma_model = roma_indoor(device, coarse_res=560, upsample_res=864)
        # self.roma_model.symmetric = False
        H, W = self.roma_model.get_output_resolution()
    
    def transform_to_tensor(self, image):
        return torch.from_numpy(image / 255.).float()[None, None].to(self.device)
    
    def __call__(self, im1_path, im2_path):
        im1, _ = read_image(im1_path)
        im2, _ = read_image(im2_path)
        H1, W1, _ = im1.shape
        H2, W2, _ = im2.shape
        self.roma_model.upsample_res = (H1, W1)
        self.model_res = (H1, W1)
        warp, certainty = self.roma_model.match(im1_path, im2_path, device=self.device)
        matches, good_certainty = self.roma_model.sample(warp, certainty, self.num_kpts)
        kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, H1, W1, H2, W2)
        return {
            "kpts1": kpts1.detach().cpu().numpy().astype(int),
            "kpts2": kpts2.detach().cpu().numpy().astype(int),
            "dense_matches": warp.detach().cpu().numpy(),
            "certainty": certainty.detach().cpu().numpy()
        }
    

