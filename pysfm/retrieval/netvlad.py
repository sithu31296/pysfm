import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import torchvision.models as models
from scipy.io import loadmat


EPS = 1e-6


class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=False)
        centers = nn.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        # From the official MATLAB implementation.
        desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc


class NetVLADModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # Create the network.
        # Remove classification head.
        backbone = list(models.vgg16().children())[0]
        # Remove last ReLU + MaxPool2d.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.netvlad = NetVLADLayer()
        self.whiten = nn.Linear(self.netvlad.output_dim, 4096)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(), weights["net"].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = weights["net"].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -weights["net"].layers[30].weights[1]  # D x K

        # Whitening weights.
        w = weights["net"].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
        b = weights["net"].layers[33].weights[1]  # Shape: OUT
        # Update layer weights.
        self.netvlad.score_proj.weight = nn.Parameter(torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1))
        self.netvlad.centers = nn.Parameter(torch.tensor(center_w).float())
        self.whiten.weight = nn.Parameter(torch.tensor(w).float().squeeze().permute([1, 0]))    # OUT x IN
        self.whiten.bias = nn.Parameter(torch.tensor(b.squeeze()).float())                      # Shape: OUT


    def forward(self, image):
        # Feature extraction.
        descriptors = self.backbone(image)
        b, c, _, _ = descriptors.size()
        descriptors = descriptors.view(b, c, -1)

        # NetVLAD layer.
        descriptors = F.normalize(descriptors, dim=1)  # Pre-normalization.
        desc = self.netvlad(descriptors)

        # Whiten if needed.
        desc = self.whiten(desc)
        desc = F.normalize(desc, dim=1)  # Final L2 normalization.

        return desc
    

class NetVLAD:
    # Models exported using
    # https://github.com/uzh-rpg/netvlad_tf_open/blob/master/matlab/net_class2struct.m.
    checkpoint_urls = {
        "VGG16-NetVLAD-Pitts30K": "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat",  # noqa: E501
        "VGG16-NetVLAD-TokyoTM": "https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat",  # noqa: E501
    }
    def __init__(self, device="cpu") -> None:
        model_name = "VGG16-NetVLAD-Pitts30K"
        # Download the checkpoint.
        checkpoint_path = Path(torch.hub.get_dir(), "netvlad", model_name + ".mat")
        if not checkpoint_path.exists():
            checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(self.checkpoint_urls[model_name], checkpoint_path)

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        weights = loadmat(checkpoint_path, struct_as_record=False, squeeze_me=True)

        self.model = NetVLADModel(weights)
        self.model = self.model.to(device)
        self.device = device
        self.mean = weights["net"].meta.normalization.averageImage[0, 0]
        self.std = np.array([1, 1, 1], dtype=np.float32)

    def __call__(self, image):
        image = image.transpose(2, 0, 1)[None, ...]
        image = torch.from_numpy(image)
        image = image - image.new_tensor(self.mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(self.std).view(1, -1, 1, 1)
        image = image.to(self.device)
        return self.model(image)