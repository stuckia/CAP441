import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

class ImgProcessing:
    def __init__(self):
        #access and store image (in grayscale)
        bicycle = os.path.join(os.path.dirname(__file__), 'bicycle.bmp')
        self.img = cv2.cvtColor(cv2.imread(bicycle), cv2.COLOR_BGR2GRAY)

        #set up subplots
        self.rows = len(self.img)
        self.cols = len(self.img[0])
        self.fig = plt.figure(figsize=(6, 9)) 

        #display original image
        self.fig.add_subplot(5,2,1)
        plt.imshow(self.img, cmap='gray')
        plt.axis('off')
        plt.title('No Filter')

        #display gaussian filter
        self.fig.add_subplot(5,2,2)
        plt.imshow(self.gaussian_filter_cv(),cmap='gray')
        plt.axis('off')
        plt.title('Gaussian Filter (OpenCV)')

        #display box filter on 3x3
        self.fig.add_subplot(5,2,3)
        plt.imshow(self.box_filter(3), cmap='gray')
        plt.axis('off')
        plt.title('Box Filter 3x3')

        #display box filter on 5x5
        self.fig.add_subplot(5,2,4)
        plt.imshow(self.box_filter(5), cmap='gray')
        plt.axis('off')
        plt.title('Box Filter 5x5')

        #display box filter on 3x3 using OpenCV
        self.fig.add_subplot(5,2,5)
        plt.imshow(self.box_filter_cv(3), cmap='gray')
        plt.axis('off')
        plt.title('Box Filter 3x3 (OpenCV)')

        #display box filter on 5x5 using OpenCV
        self.fig.add_subplot(5,2,6)
        plt.imshow(self.box_filter_cv(5), cmap='gray')
        plt.axis('off')
        plt.title('Box Filter 5x5 (OpenCV)')

        #display sobel filter on x
        self.fig.add_subplot(5,2,7)
        plt.imshow(self.x_sobel_filter(), cmap='gray')
        plt.axis('off')
        plt.title('Sobel X Filter')

        #display sobel filter on y
        self.fig.add_subplot(5,2,8)
        plt.imshow(self.y_sobel_filter(), cmap='gray')
        plt.axis('off')
        plt.title('Sobel Y Filter')

        #display combined sobel filter
        self.fig.add_subplot(5,2,9)
        plt.imshow(self.sobel_filter(), cmap='gray')
        plt.axis('off')
        plt.title('Combined Sobel Filter')

        #display combined sobel filter using OpenCV
        self.fig.add_subplot(5,2,10)
        plt.imshow(self.sobel_filter_cv(), cmap='gray')
        plt.axis('off')
        plt.title('Combined Sobel Filter (OpenCV)')

        plt.show()

    # get sum of 3x3 matrix values
    def matrix_sum(self, sq, size):
        sq_sum=0

        for i in range(size):
            for j in range(size):
                sq_sum+=sq[i][j]

        return (sq_sum // (size*size))

    #box filter, no cv
    def box_filter(self, size):
        new_img = self.img

        curr_row = 0
        curr_col = 0

        sq = []
        sq_row = []
        filter_img = []
        filter_row = []

        mod = size

        while curr_row <= self.rows - mod:
            while curr_col <= self.cols - mod:
                for i in range(curr_row, curr_row+mod):
                    for j in range(curr_col, curr_col+mod):
                        sq_row.append(new_img[i][j])
                    sq.append(sq_row)
                    sq_row = []
                filter_row.append(self.matrix_sum(sq, size))
                sq = []
                curr_col += 1
            filter_img.append(filter_row)
            filter_row = []
            curr_row += 1
            curr_col = 0

        return filter_img

    #box filter, cv
    def box_filter_cv(self, size):
        new_img = cv2.blur(self.img, (size,size))

        return new_img

    #sobel filter towards x-axis edges, no cv
    def x_sobel_filter(self):
        filter_img = np.copy(self.img)

        for i in range(1,self.rows-1):
            for j in range(1,self.cols-1):
                x_pos = (self.img[i-1][j-1] + 2*self.img[i][j-1] + self.img[i+1][j-1])
                x_neg = (self.img[i-1][j+1] + 2*self.img[i][j+1] + self.img[i+1][j+1])
                filter_img[i][j] = min(255, (x_pos-x_neg))

        return filter_img

    #sobel filter towards y-axis edges, no cv
    def y_sobel_filter(self):
        filter_img = np.copy(self.img)

        for i in range(1,self.rows-1):
            for j in range(1,self.cols-1):
                y_pos = (self.img[i-1][j-1] + 2*self.img[i-1][j] + self.img[i-1][j+1])
                y_neg = (self.img[i+1][j-1] + 2*self.img[i+1][j] + self.img[i+1][j+1])
                filter_img[i][j] = min(255, (y_pos-y_neg))

        return filter_img

    #sobel filter with x- and y-axis edges, no cv
    def sobel_filter(self):
        filter_img = np.copy(self.img)

        for i in range(1,self.rows-1):
            for j in range(1,self.cols-1):
                x_pos = (self.img[i-1][j-1] + 2*self.img[i][j-1] + self.img[i+1][j-1])
                x_neg = (self.img[i-1][j+1] + 2*self.img[i][j+1] + self.img[i+1][j+1])
                y_pos = (self.img[i-1][j-1] + 2*self.img[i-1][j] + self.img[i-1][j+1])
                y_neg = (self.img[i+1][j-1] + 2*self.img[i+1][j] + self.img[i+1][j+1])
                filter_img[i][j] = min(255, np.sqrt((x_pos-x_neg)**2 + (y_pos-y_neg)**2))

        return filter_img

    #sobel filter with x- and y-axis edges, cv
    def sobel_filter_cv(self):
        sobel_x = cv2.convertScaleAbs(cv2.Sobel(self.img, cv2.CV_64F,1,0,ksize=3))
        sobel_y = cv2.convertScaleAbs(cv2.Sobel(self.img, cv2.CV_64F,0,1,ksize=3))

        return cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
        
    #gaussian filter, cv
    def gaussian_filter_cv(self):
        return cv2.GaussianBlur(self.img, (3,3), 0)

if __name__ == '__main__':
    # call class to initiate program
    p = ImgProcessing()