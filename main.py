import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

class ImgHistogram:
    # constructor
    def __init__(self):
        self.img = cv2.imread('dog.bmp')
        self.img_prev = cv2.imread('dog.bmp')
        self.new_img = cv2.imread('dog.bmp')
        
        rows = 2
        columns = 2
        self.fig, self.axes=plt.subplots(nrows=2,ncols=2, figsize=(columns*3,rows*3))
        self.fig.subplots_adjust(left=0.25, bottom=0.25)
        self.fig.add_subplot(rows,columns,1)
        plt.axis('off')        
        plt.imshow(self.img, aspect='auto')
        
        self.fig.add_subplot(rows,columns,2)
        plt.axis('off')
        plt.imshow(self.img_prev, aspect='auto')
        
        cols = ['Original Image','Altered Image']
        for ax,col in zip(self.axes[0],cols):
            ax.set_title(col)
        for i,ax in enumerate(self.axes.flat):
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.fig.add_subplot(rows,columns,3)
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.img],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])
        
        self.fig.add_subplot(rows,columns,4)
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.img_prev],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])
        
        # slider bar axes
        axcontrast = plt.axes([0.25, 0.1, 0.65, 0.03])
        axbrightness = plt.axes([0.25, 0.15, 0.65, 0.03])
        axbutton = plt.axes([0.525,0.05,0.1,0.03])
        
        #contrast can go from 0<1 for decrease, >1 for increase
        self.contrast = Slider(axcontrast, 'Contrast', 0.0, 2.0, valinit=1.0)
        #brightness can go from -127 to 127
        self.brightness = Slider(axbrightness, 'Brightness', -100.0, 100.0, valinit=0.0)
        self.savebutton = Button(axbutton, 'Save')

        self.contrast.on_changed(self.update)
        self.brightness.on_changed(self.update)
        self.savebutton.on_clicked(self.saveimg)
        
        plt.show()

        cv2.waitKey(0)
    
    def update(self,val):
        c = self.contrast.val
        b = self.brightness.val
        self.new_img = cv2.addWeighted(self.img_prev, c, self.img_prev,0,b)

        plt.subplot(2,2,2)
        plt.axis('off')
        plt.imshow(self.new_img, aspect='auto')

        plt.subplot(2,2,4)
        plt.cla()
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.new_img],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])

        self.fig.canvas.draw()

    def saveimg(self,val):
        if not cv2.imwrite(r'C:\Users\itsme\Documents\CAP4410\dog.bmp', self.new_img):
            raise Exception("Could not write image")


if __name__ == '__main__':
    h = ImgHistogram()

# add save functionality of new image
# when reload code, show new image in original image location (replace)