from pysfm import *



class FLANN:
    def __init__(self) -> None:
        self.ratio_threshold = 0.8 
        # self.ransac_threshold = 10
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, des1, des2):
        matches = self.matcher.knnMatch(des1, des2, k=2)
        match_indices = []
        for m, n in matches:    # ratio test as per Lowe's paper
            if m.distance < self.ratio_threshold * n.distance:
                match_indices.append([m.queryIdx, m.trainIdx])
        match_indices = np.array(match_indices)
        return match_indices



# class FLANN:
#     def __init__(self) -> None:
#         self.ratio_threshold = 0.8 
#         # self.ransac_threshold = 10
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)
#         self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
#     def match(self, des1, des2):
#         matches = self.matcher.knnMatch(des1, des2, k=2)
#         match = []
#         for m, n in matches:    # ratio test as per Lowe's paper
#             if m.distance < self.ratio_threshold * n.distance:
#                 match.append([m.queryIdx, m.trainIdx])
#         pts1, pts2 = np.array(pts1), np.array(pts2)
#         match = np.array(match)
#         print(match.shape)

#         # # constrain matches to fit homography
#         # retval, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_threshold)
#         # mask = mask.ravel() # len(pts1)

#         # # select inlier points
#         # pts1, pts2 = pts1[mask == 1], pts2[mask == 1]
#         return pts1, pts2
    

class RANSAC:
    def __init__(self):
        # RANSAC parameters
        self.ransac_threshold = 60
        self.ransac_iters = 200

    def homography(self, pts0, pts1):
        rows = []
        for i in range(pts0.shape[0]):
            p1 = np.append(pts0[i], 1)
            p2 = np.append(pts1[i], 1)
            row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
            row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
            rows.append(row1)
            rows.append(row2)
        rows = np.array(rows)
        U, s, V = np.linalg.svd(rows)
        H = V[-1].reshape(3, 3)
        H = H/H[2, 2] # standardize to let w*H[2,2] = 1
        return H
    
    def get_error(self, pts0, pts1, H):
        num_points = len(pts0)
        all_p1 = np.concatenate((pts0, np.ones((num_points, 1))), axis=1)
        all_p2 = pts1
        estimate_p2 = np.zeros((num_points, 2))
        for i in range(num_points):
            temp = np.dot(H, all_p1[i])
            estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
        # Compute error
        errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2
        return errors
    
    def random_point(self, pts0, pts1, k=4):
        idx = random.sample(range(len(pts0)), k)
        pts0 = [pts0[i] for i in idx]
        pts1 = [pts1[i] for i in idx]
        return np.array(pts0), np.array(pts1)

    def __call__(self, kpts0, kpts1): 
        num_best_inliers = 0
        best_kpts0, best_kpts1 = kpts0, kpts1
        for i in range(self.ransac_iters):
            pts0, pts1 = self.random_point(kpts0, kpts1)
            H = self.homography(pts0, pts1)
            if np.linalg.matrix_rank(H) < 3:    # avoid dividing by zero
                continue

            errors = self.get_error(kpts0, kpts1, H)
            idx = np.where(errors < self.ransac_threshold)[0]
            inliers_pts0, inliers_pts1 = kpts0[idx], kpts1[idx]
            num_inliers = len(inliers_pts0)
            if num_inliers > num_best_inliers:
                best_kpts0, best_kpts1 = inliers_pts0, inliers_pts1
                num_best_inliers = num_inliers
        return best_kpts0, best_kpts1



# def main(img0_path, img1_path):
#     im0 = read_image(img0_path, None)
#     im1 = read_image(img1_path, (1920, 1080))

#     matcher = SIFTMatching()
#     ransac = RANSAC()
#     data = {"image0": im0, "image1": im1}
#     pts1, pts2 = matcher.get_correspondences(data)
#     draw_matches(pts1, pts2, im0, im1)
#     pts1, pts2 = ransac(pts1, pts2)
    


if __name__ == '__main__':
    img0_path = "assets/test/31.png"
    img1_path = "assets/test/IMG_8267.jpg"

    # main(img0_path, img1_path)