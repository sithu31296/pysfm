from pysfm import *
from uniflowmatch.models.ufm import UniFlowMatchConfidence, UniFlowMatchClassificationRefinement
from pysfm.utils.imageio import read_image
import flow_vis

def warp_image_with_flow(source_image, source_mask, target_image, flow) -> np.ndarray:
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.

    Args:
    source_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    target_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    flow: np.ndarray of shape (H, W, 2)
    source_mask: non_occluded mask represented in source image.

    Returns:
    warped_image: target_image warped according to flow into frame of source image
    np.ndarray of shape (H, W, 3), normalized to [0, 1]

    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > 0.5)

    return warped_image


def visualize_results(source_image, target_image, flow_output, covisibility, output_path="ufm_output.png"):
    """Create and save visualization of results."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Input images and warped result
    axs[0, 0].imshow(source_image)
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(target_image)
    axs[0, 1].set_title("Target Image")
    axs[0, 1].axis("off")

    # Warp the image using flow
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0))
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
        warped_image
    )
    warped_image = np.clip(warped_image / 255.0, 0, 1)

    axs[0, 2].imshow(warped_image)
    axs[0, 2].set_title("Warped Source Image")
    axs[0, 2].axis("off")

    # Bottom row: Flow and covisibility visualizations
    flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    axs[1, 0].imshow(flow_vis_image)
    axs[1, 0].set_title("Flow Visualization (Valid at Covisible Pixels)")
    axs[1, 0].axis("off")

    # Covisibility mask (thresholded)
    axs[1, 1].imshow(covisibility > 0.5, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Covisibility Mask (>0.5)")
    axs[1, 1].axis("off")

    # Covisibility mask (continuous)
    heatmap = axs[1, 2].imshow(covisibility, cmap="viridis", vmin=0, vmax=1)
    axs[1, 2].set_title("Covisibility Confidence")
    axs[1, 2].axis("off")
    plt.colorbar(heatmap, ax=axs[1, 2], shrink=0.6)

    plt.tight_layout()
    # plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # print(f"Visualization saved to: {output_path}")
    plt.show()

    return fig

class UFM:
    def __init__(self, weights_path=None, device="cpu", refine=False, num_kpts=10000) -> None:
        self.device = device
        self.num_kpts = num_kpts
        self.model_res = (864, 864)
        if refine:
            self.model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base")
        else:
            self.model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
        self.model.eval()


    def transform_to_tensor(self, image):
        return torch.from_numpy(image / 255.).float()[None, None].to(self.device)
    
    def __call__(self, im1_path, im2_path):
        im1, _ = read_image(im1_path)
        im2, _ = read_image(im2_path)
        H1, W1, _ = im1.shape
        H2, W2, _ = im2.shape

        with torch.inference_mode():
            result = self.model.predict_correspondences_batched(
                source_image=torch.from_numpy(im1),
                target_image=torch.from_numpy(im2),
            )
            flow = result.flow.flow_output[0].cpu().numpy()
            covisibility = result.covisibility.mask[0].cpu().numpy()

        kpts1, kpts2, warp = self.flow_to_kpts(im1, im2, flow.transpose(1, 2, 0), covisibility)

        # certainty = np.concatenate([covisibility, covisibility], axis=1)
        certainty = covisibility

        return {
            "kpts1": kpts1.astype(int),
            "kpts2": kpts2.astype(int),
            "dense_matches": warp.astype(np.float32),
            "certainty": certainty.astype(np.float32)
        }
    

    def flow_to_kpts(self, im1, im2, flow, covisibility):
        assert flow.shape[-1] == 2

        height, width = im1.shape[:2]
        target_height, target_width = im2.shape[:2]

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # apply flow displacements
        flow_x, flow_y = flow[..., 0], flow[..., 1]
        
        x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
        y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

        kpts1 = np.concatenate([x[..., None], y[..., None]], axis=-1)
        kpts2 = np.concatenate([x_new[..., None], y_new[..., None]], axis=-1)

        x_norm = (x / width) * 2 - 1
        y_norm = (y / height) * 2 - 1
        x_new_norm = (x_new / target_width) * 2 - 1
        y_new_norm = (y_new / target_height) * 2 - 1

        kpts1_norm = np.concatenate([x_norm[..., None], y_norm[..., None]], axis=-1)
        kpts2_norm = np.concatenate([x_new_norm[..., None], y_new_norm[..., None]], axis=-1)

        warp = np.concatenate([kpts1_norm, kpts2_norm], axis=-1)
        mask = (covisibility > 0.5).reshape(-1)

        return kpts1.reshape(-1, 2)[mask], kpts2.reshape(-1, 2)[mask], warp