# import the necessary packages
from __future__ import print_function
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.165


def find_crosshairs(image):
    # load image and convert it to grayscale
   # img_rgb = image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load template
    template = cv2.imread('20191107_201153.jpg', 0)
    w, h = template.shape[::-1]
    template = cv2.resize(template, (int(h/10), int(w/10)))

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # similarity threshold
    threshold = 0.16
    loc = np.where(res >= threshold)

    list_of_centers = []
    list_of_centers2 = []
    x_coor = []
    n = 0

    for pt in zip(*loc[::-1]):  # loop over all the templates found in image
        x_center, y_center = pt[0] + int(w / 2), pt[1] + int(h / 2)
        x_coor.append(pt[0] + int(w / 2))
        crosshair_center = tuple([x_center, y_center])

        if n == 0:
            list_of_centers.append(crosshair_center)

            #cv2.circle(image, crosshair_center, int(h/2.5), (0, 255, 255), 2)

        elif n >= 1 and x_coor[n - 1] < x_coor[n] - 50 or x_coor[n - 1] > x_coor[
            n] + 50:  # filters our similar coordinates

            list_of_centers.append(crosshair_center)
            list_of_centers2.append(crosshair_center)
            #cv2.circle(image, crosshair_center, w, (0, 255, 255), 2)
        n = n + 1

    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
   # print(list_of_centers)

    #img_rgb = cv2.resize(img_rgb, (500, 500))
    #cv2.imshow('Template', template)
    #cv2.imshow('Detected Targets', image)

    #use the width (in pixels) of the detected target to find our real world pixel/inch ratio
    pix_per_inch = w/2
    return list_of_centers, pix_per_inch


def alignImages(imB, imA):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(imB, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(imA, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(imB, keypoints1, imA, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = imA.shape
    im1Reg = cv2.warpPerspective(imB, h, (width, height))

    return im1Reg, h

def resizeImages(IMA, IMB, r):
#find size of images
    heightA, widthA, channelA = IMA.shape
    heightB, widthB, channelB = IMB.shape
    #print(heightA,widthA,channelA,heightB,widthB,channelB)

    #resize image 1
    heightA = int(heightA/ r)
    widthA = int(widthA/ r)
    IMA= cv2.resize(IMA, (heightA, widthA))

    #resize image2
    IMB = cv2.resize(IMB, (heightA, widthA))

    return IMA, IMB




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
                help="first input image")
ap.add_argument("-s", "--second", required=True,
                help="second")
args = vars(ap.parse_args())


# load the two input images
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

imageB, h = alignImages(imageB, imageA)
imageA, imageB = resizeImages(imageA, imageB, 1)


#Align and Resize Images
times = 3
i=1

while i < times:
    imageB , h = alignImages(imageB, imageA)
    imageA, imageB = resizeImages(imageA, imageB, 1)
    i +=1

#find targets and their coordinates
imageC = imageA.copy()
crosshair_center = []
crosshair_center, pixelsperinch = find_crosshairs(imageC)
print(crosshair_center)
i=1
for xy in crosshair_center:
    cv2.circle(imageC, xy, 100, (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(imageC, str(i), xy, font, 1, (200, 255, 155), 2, cv2.LINE_AA)
    i=i+1
cv2.imshow('Pick a Target', imageC)
cv2.waitKey(0)

#receive input of which target user will be shooting at
target = input('Which target will you be shooting at?')

# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#find avg area of found differences in order to use for filtering effects
areas = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    wa, ha, c = imageA.shape
    picture_area = wa*ha
    area = w*h
    if area < picture_area/4 : #this filters out large area differences
        areas.append(area)

# loop over the contours
i=0
avg_area = sum(areas) / (len(areas)+1) #avg area of differences (not including large differences)
print('average area: ')
print(avg_area)
#print(avg_area)
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
 #we need to filter out some differences
    area = w*h
    heightA, widthA, channelA = imageA.shape
    x_center = widthA / 2
    y_center = heightA / 2
    if  area > avg_area/2 and area < picture_area/10: #this filters out the micro differences and more large differences area > avg_area/2 and
        if w < h*1.4 and w > h*.6: #this filters out irregular shapes (non-squares)
            if x > 0 + widthA/6 and x < widthA-widthA/6 and y > 0 + heightA/6 and y < heightA-heightA/6: #filters out edges
                i=i+1
                cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
                shot_coor_x = x
                shot_coor_y = y
                horizontal_diff = (x - crosshair_center[1][0]) * pixelsperinch**-1 #positive means right
                vertical_diff = (crosshair_center[1][1] - y) * pixelsperinch**-1 #positive means high
                cv2.line(imageB, (x, y), (crosshair_center[1]), (0, 255, 0), 2)
                #print('area: ')
                #print(area)
                print('too high and too right have positive values')
                print('up or down (inches): ' + str(vertical_diff))
                print('left or right (inches) ' + str(horizontal_diff))
                #print(shot_coor_x, shot_coor_y)
                #print(horizontal_diff)
                #print(vertical_diff)

i = str(i)
for xy in crosshair_center:
    cv2.circle(imageB, xy, 100, (0, 255, 255), 2)
print('number of matches: ' + i)
# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
#cv2.imshow("Diff", diff)
#cv2.imshow("Thresh", thresh)
cv2.waitKey(0)


