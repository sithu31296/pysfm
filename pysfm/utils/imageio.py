from pysfm import *
import pillow_heif
from PIL import ExifTags, Image, TiffTags
from pillow_heif import register_heif_opener

register_heif_opener()

img_extensions = ['.png', '.jpg', '.jpeg', 'heif', 'heic']


def extract_exif(img_pil: Image) -> Dict[str, Any]:
    """Return exif information as a dictionary.

    Args:
    ----
        img_pil: A Pillow image.

    Returns:
    -------
        A dictionary with extracted EXIF information.

    """
    # Get full exif description from get_ifd(0x8769):
    # cf https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    tiff_tags = img_pil.getexif()
    tiff_dict = {
        TiffTags.TAGS_V2[k].name: v
        for k, v in tiff_tags.items()
        if k in TiffTags.TAGS_V2
    }
    return {**exif_dict, **tiff_dict}


def fpx_from_f35(width: float, height: float, f_mm: float = 50) -> float:
    """Convert a focal length given in mm (35mm film equivalent) to pixels."""
    return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def rgb2gray(img: Union[np.ndarray, torch.Tensor]):
    """img in shape [C, H, W] in rgb format"""
    img = img.squeeze()
    if isinstance(img, torch.Tensor):   
        img = img.permute(1, 2, 0).numpy()
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def tensor_to_image(x: Union[Tensor, Image.Image, np.ndarray]):
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return x
    if x.max() <= 1.01:
        x *= 255
    if x.shape[0] == 3:
        x = x.transpose((1, 2, 0))
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def read_image(path: Union[str, Path], auto_rotate: bool = True, remove_alpha: bool = True):
    """Read the image and also extract EXIF data if exists
    Args:
    ----
        path: The url to the image to load.
        auto_rotate: Rotate the image based on the EXIF data, default is True.
        remove_alpha: Remove the alpha channel, default is True.

    Returns:
    -------
        img: The image loaded as a numpy array.
        icc_profile: The color profile of the image.
        f_px: The optional focal length in pixels, extracting from the exif data.
    """
    path = Path(path)
    if path.suffix.lower() in [".heic"]:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(path)

    img_exif = extract_exif(img_pil)
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)

    img = np.array(img_pil)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    # Extract the focal length from exif data.
    f_35mm = img_exif.get("FocalLengthIn35mmFilm",
        img_exif.get("FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)),
    )
    if f_35mm is not None and f_35mm > 0:
        # print(f"\tfocal length @ 35mm film: {f_35mm}mm")
        f_px = fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    else:
        f_px = None
    return img, f_px


def read_image_paths(folder: Union[str, Path]):
    folder = Path(folder)
    image_paths = sorted([x for x in folder.iterdir() if x.suffix.lower() in img_extensions], key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    return image_paths


def read_images(path: Union[str, Path, list], *args, **kwargs):
    if isinstance(path, list):
        image_paths = path
        if isinstance(image_paths[0], np.ndarray):
            return image_paths
    else:
        path = Path(path)
        if path.is_dir():
            image_paths = read_image_paths(path)
        elif path.is_file():
            image_paths = [path]
    
    images, focals = [], []
    for image_path in image_paths:
        image, focal = read_image(image_path, *args, **kwargs)
        images.append(image)
        focals.append(focal)
    return images, focals
