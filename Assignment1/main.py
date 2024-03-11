import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

class ImgHistogram:
    # constructor
    def __init__(self):
        # read in images
        dog = os.path.join(os.path.dirname(__file__), 'dog.bmp')
        self.img = cv2.imread(dog)
        self.img_prev = self.img
        
        # create initial subplots
        rows = 2
        columns = 2
        self.fig, self.axes=plt.subplots(nrows=2,ncols=2, figsize=(columns*3,rows*3))
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.25)

        # add subplot for original image
        self.fig.add_subplot(rows,columns,1)
        plt.axis('off')        
        plt.imshow(self.img, aspect='auto')
        
        # add subplot for altered image
        self.fig.add_subplot(rows,columns,2)
        plt.axis('off')
        plt.imshow(self.img_prev, aspect='auto')
        
        # adjust vertices and axes of subplots for visual enhancement
        cols = ['Original Image','Altered Image']
        for ax,col in zip(self.axes[0],cols):
            ax.set_title(col)
        for i,ax in enumerate(self.axes.flat):
            ax.set_xticks([])
            ax.set_yticks([])

        # add subplot for original histogram   
        self.fig.add_subplot(rows,columns,3)
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.img],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])
        
        # add subplot for altered histogram
        self.fig.add_subplot(rows,columns,4)
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.img_prev],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])
        
        # declare slider bar axes
        axcontrast = plt.axes([0.25, 0.1, 0.65, 0.03])
        axbrightness = plt.axes([0.25, 0.15, 0.65, 0.03])
        
        # contrast can go from 0<1 for decrease, >1 for increase
        self.contrast = Slider(axcontrast, 'Contrast', 0.0, 2.0, valinit=1.0)
        # brightness can go from -127 to 127
        self.brightness = Slider(axbrightness, 'Brightness', -100.0, 100.0, valinit=0.0)

        # set on_changed to trigger update function
        self.contrast.on_changed(self.update)
        self.brightness.on_changed(self.update)
        
        # declare button axes
        axbutton = plt.axes([0.525,0.05,0.1,0.03])

        # initialize button
        self.savebutton = Button(axbutton, 'Save')

        # set on_clicked to trigger saveimg function
        self.savebutton.on_clicked(self.saveimg)

        # show plots in the window
        plt.show()

        # wait for exit
        cv2.waitKey(0)
    
    # update altered image and histogram
    def update(self,val):
        # change image values with slider usage
        self.new_img = cv2.addWeighted(self.img_prev, self.contrast.val, self.img_prev,0,self.brightness.val)

        # update altered image subplot to display changes
        plt.subplot(2,2,2)
        plt.axis('off')
        plt.imshow(self.new_img, aspect='auto')

        # update altered image histogram subplot to display changes
        plt.subplot(2,2,4)
        plt.cla()
        color = ('b','g','r')
        for i,col in enumerate(color):
            imghist = cv2.calcHist([self.new_img],[i],None,[256],[0,256])
            plt.plot(imghist,color = col)
            plt.xlim([0,256])

        # display changes on the plot
        self.fig.canvas.draw()

    # save altered image in place of the original image
    def saveimg(self,val):
        # raise exception if code cannot write to file
        if not cv2.imwrite(r'C:\Users\itsme\Documents\CAP4410\Assignment1\dog.bmp', self.new_img):
            raise Exception("Could not write image to file")

if __name__ == '__main__':
    # call class to initiate program
    h = ImgHistogram()