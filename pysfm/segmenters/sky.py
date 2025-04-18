from pysfm import *
import copy
import onnxruntime
from pysfm.utils.utils import download_file_from_url
from pysfm.utils.imageio import read_images



def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")
    return onnx_result



class SkyRemover:
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """
    def __init__(self):
        ckpt_dir = Path("./checkpoints")
        sky_model_path = ckpt_dir / "skyseg.onnx"

        if not sky_model_path.exists():
            download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", sky_model_path)
        
        self.model_session = onnxruntime.InferenceSession(sky_model_path)

    
    def __call__(self, path: Union[str, list, Path]):
        sky_masks = []
        images = read_images(path)[0]
        for image in images:
            result_map = run_skyseg(self.model_session, [320, 320], image)
            # resize the result_map to the original image size
            result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

            # Fix: Invert the mask so that 255 = non-sky, 0 = sky
            # The model outputs low values for sky, high values for non-sky
            output_mask = np.zeros_like(result_map_original)
            output_mask[result_map_original < 32] = 255  # Use threshold of 32
            sky_masks.append(output_mask.astype(bool))
        
        if not isinstance(path, list):
            if Path(path).is_file():
                return sky_masks[0]
        return sky_masks