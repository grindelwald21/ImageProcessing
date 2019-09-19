import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test_image2.jpg')
lane_image = np.copy(image)

gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Gray",gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

# cv2.imshow("Blured",blur)

canny = cv2.Canny(blur, 50, 150)


# cv2.imshow("canny",canny)

def region_of_intrest(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return mask


# plt.imshow(region_of_intrest(canny))
# plt.show()

masked_image = cv2.bitwise_and(canny,region_of_intrest(canny))
# plt.imshow(masked_image)
# plt.show()


'''
implementation of the Hough Transform
'''

lines = cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength=130,maxLineGap=5)

# image: the image that we want to detect the lines in
# rth: The resolution of the parameter r in pixels. We use 2 pixels
# theta: The resolution of the parameter θ in radians. We use 1 degree but we should use the radians
# threshold: The minimum number of intersections to "*detect*" a line
# lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
# minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
# maxLineGap: The maximum gap between two points to be considered in the same line.

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

line_image = display_lines(lane_image,lines)
combo = cv2.addWeighted(lane_image,0.8,line_image,1,1)
cv2.imshow("t", combo)

cv2.waitKey(0)
