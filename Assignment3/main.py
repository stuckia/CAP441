import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_multiotsu

class Segmentation:
    def __init__(self):
        # read input data
        img1 = os.path.join(os.path.dirname(__file__), 'OTSU2class-edge_L-150x150.png')
        otsu_img1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
        otsu_img1_gray = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)

        img2 = os.path.join(os.path.dirname(__file__), 'OTSU2class-andreas_L-150x150.png')
        otsu_img2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
        otsu_img2_gray = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        img3 = os.path.join(os.path.dirname(__file__), 'OTSU_Multiple_Class-S01-150x150.png')
        otsu_img3 = cv2.cvtColor(cv2.imread(img3), cv2.COLOR_BGR2RGB)
        otsu_img3_gray = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)

        img4 = os.path.join(os.path.dirname(__file__), 'meanshift_S00-150x150.png')
        mean_img = cv2.cvtColor(cv2.imread(img4), cv2.COLOR_BGR2RGB)
        mean_img_gray = cv2.imread(img4, cv2.IMREAD_GRAYSCALE)


        # create figure and axes for displayal
        fig1, axes1 = plt.subplots(4, 3)
        fig1.tight_layout()


        # otsu binarization with two pixel classes
        _, thresh1 = cv2.threshold(otsu_img1_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hist1 = cv2.calcHist([otsu_img1_gray], [0], None, [256], [0, 256])


        # display first set of images and matching histogram
        axes1[0][0].set_title('Original Image')
        axes1[0][0].axis('off')
        axes1[0][0].imshow(otsu_img1, cmap='gray')
        axes1[0][1].set_title('Histogram')
        axes1[0][1].plot(hist1)
        axes1[0][2].set_title('Otsu\'s Binarization')
        axes1[0][2].axis('off')
        axes1[0][2].imshow(thresh1, cmap='gray')


        # otsu binarization with two pixel classes
        _, thresh2 = cv2.threshold(otsu_img2_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hist2 = cv2.calcHist([otsu_img2_gray], [0], None, [256], [0, 256])


        # display second set of images and matching histogram
        axes1[1][0].set_title('Original Image')
        axes1[1][0].axis('off')
        axes1[1][0].imshow(otsu_img2, cmap='gray')
        axes1[1][1].set_title('Histogram')
        axes1[1][1].plot(hist2)
        axes1[1][2].set_title('Otsu\'s Binarization')
        axes1[1][2].axis('off')
        axes1[1][2].imshow(thresh2, cmap='gray')


        # otsu binarization with multiple classes
        thresh = threshold_multiotsu(otsu_img3_gray)
        reg = np.digitize(otsu_img3_gray, bins=thresh)


        # display third set of images and matching histogram
        axes1[2][0].set_title('Original Image')
        axes1[2][0].axis('off')
        axes1[2][0].imshow(otsu_img3, cmap='gray')
        axes1[2][1].set_title('Histogram')
        axes1[2][1].hist(otsu_img3_gray.ravel(), 255)
        for t in thresh:
            axes1[2][1].axvline(t, color='r')
        axes1[2][2].set_title('Otsu\'s Binarization (Multi)')
        axes1[2][2].axis('off')
        axes1[2][2].imshow(reg, cmap='gray')


        # mean shift method
        img_rows, img_cols = mean_img_gray.shape
        mean_shift = np.zeros_like(mean_img_gray)
        radius = 3
        band = 1
        for r in range(img_rows):
            for c in range(img_cols):
                cluster = mean_img_gray[max(0, r-radius):min(img_rows, r+1+radius),
                                        max(0, c-radius):min(img_cols, c+1+radius)]
                vector = np.mean(cluster, axis=(0,1)) - mean_img_gray[r,c]
                mean_shift[r,c] = mean_img_gray[r,c] + vector * band

        #display fourth set of images and matching histogram
        axes1[3][0].set_title('Original Image')
        axes1[3][0].axis('off')
        axes1[3][0].imshow(mean_img)
        axes1[3][1].set_title('Histogram')
        axes1[3][1].hist(mean_shift.ravel(), 255)
        axes1[3][2].set_title('Mean Shift')
        axes1[3][2].axis('off')
        axes1[3][2].imshow(mean_shift, cmap='gray')

        plt.show()

if __name__ == '__main__':
    # call class to initiate program
    s = Segmentation()