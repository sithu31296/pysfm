from pysfm import *
from pysfm.utils.imageio import rgb2gray, load_images


EPS = 1e-7

class SIFT:
    def __init__(self) -> None:
        # SIFT parameters 
        self.min_num_kpts = 4
        self.num_descriptors = 1024     # None -> detect as much as it can
        self.descriptor = cv2.SIFT_create(self.num_descriptors)
    
    def root_sift(self, descs):
        """Apply the Hellinger kernel by first L1-normalizing, taking the square-root, and then l2-normalizing"""
        descs /= (descs.sum(axis=1, keepdims=True) + EPS)
        descs = np.sqrt(descs)
        return descs
    
    def __call__(self, images, image_paths):
        new_images, new_image_paths = [], []
        keypoints, descriptors, colors = [], [], []
        for i, image in enumerate(images):
            gray = rgb2gray(image)
            kpts, des = self.descriptor.detectAndCompute(gray, None) # (#num_kpts), (#num_kpts, num_feats)
            des = self.root_sift(des)   # apply normalization (rootSIFT)
            if len(kpts) <= self.min_num_kpts:
                continue
            keypoint, color = [], []
            for kpt in kpts:
                pt = kpt.pt # (x, y) or (w, h)
                keypoint.append(pt)
                color.append(image[int(pt[1]), int(pt[0])])
            
            keypoints.append(np.array(keypoint))
            colors.append(np.array(color))
            descriptors.append(des)
            new_images.append(image)
            new_image_paths.append(image_paths[i])
        return np.array(keypoints), np.array(colors), np.array(descriptors), new_images, new_image_paths


if __name__ == '__main__':
    image_folder = "assets/statue"
    images, image_paths = load_images(image_folder)
    descriptor = SIFT()
    kpts, colors, des, images, image_paths = descriptor.get_descriptors(images, image_paths)
    print(kpts.shape, colors.shape, des.shape)