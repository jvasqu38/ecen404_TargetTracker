# import the necessary packages
from __future__ import print_function
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

import numpy as np

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

#find size of images
heightA, widthA, channelA = imageA.shape
heightB, widthB, channelB = imageA.shape
print(heightA,widthA,channelA,heightB,widthB,channelB)

#resize image 1
heightA = int(heightA/3)
widthA = int(widthA/3)
#imageA = cv2.resize(imageA, (heightA, widthA))

#resize image2
#imageB = cv2.resize(imageB, (heightA, widthA))



#ALIGNIMAGES
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.03

# Convert images to grayscale
im1Gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)

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
imMatches = cv2.drawMatches(imageB, keypoints1, imageA, keypoints2, matches, None)
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
height, width, channels = imageA.shape
imageB = cv2.warpPerspective(imageB, h, (width, height))

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

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
 #we need to filter out some of the images
    area = w * h
   # if area > 2500 and area < 3500:
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    shot_coor_x = x
    shot_coor_y = y
    print(shot_coor_x,shot_coor_y)

imageA = cv2.resize(imageA, (heightA, widthA))

#resize image2
imageB = cv2.resize(imageB, (heightA, widthA))
diff = cv2.resize(diff, (heightA, widthA))
thresh = cv2.resize(thresh, (heightA, widthA))
# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)