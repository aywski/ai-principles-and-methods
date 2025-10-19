import cv2
from skimage.feature import hog

def load_image_hog(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 96))
    features = hog(img, pixels_per_cell=(32,32),
                   cells_per_block=(2,2),
                   orientations=9,
                   block_norm='L2-Hys')
    return features