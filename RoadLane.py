import cv2
import numpy as np
import matplotlib.pyplot as plt

def RoI(img,vertices):
    mask =np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask , vertices , match_mask_color)
    masked_images = cv2.bitwise_and(img,mask)
    return masked_images

def draw_line(img,lines):
    cimg = np.copy(img)
    bimg = np.zeros((cimg.shape[0],cimg.shape[1],3),dtype = np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(bimg,(x1,y1),(x2,y2),(0,255,255),3)
            
    img = cv2.addWeighted(img,0.8,bimg,1,0.0)
    return img
    
image = cv2.imread('hough.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]

roi_vertices = [
        (0,height),
        (width/2,height/2),
        (width,height)]


gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cannyimg =  cv2.Canny(gray,110,200)
cropped_image = RoI(cannyimg,np.array([roi_vertices],np.int32),)

lines = cv2.HoughLinesP(cropped_image,rho = 4, theta = np.pi/60,threshold = 160,lines = np.array([]),minLineLength = 40,maxLineGap = 25)

img_with_lines = draw_line(image,lines)



plt.imshow(img_with_lines)
plt.show()


#cv2.imshow('frame',cropped_image)

#cv2.waitKey(0)
cv2.destroyAllWindows()
