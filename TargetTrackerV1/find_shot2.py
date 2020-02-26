# import the necessary packages
from __future__ import print_function
from skimage.measure import compare_ssim
from skimage.transform import resize
import argparse
import imutils
import cv2
import statistics
import numpy as np
import settings
settings.init()

MAX_FEATURES = 500  # global constant used in alignImages function
GOOD_MATCH_PERCENT = 0.18  # global constant used in alignImages function



# this function is used to find the pre-drawn crosshairs in our images.
def find_crosshairs(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convertion to black and gray important for finding shapes
    img = cv2.resize(img, (560, 1020), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # will create noise in our image to better pick up shapes
    # img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # to find our crosshairs, we will be using CV function HoughCircles.
    # this functions finds circles in an image and returns their pixel coordinates and the radius
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=20, maxRadius=50)

    circles = np.uint16(np.around(circles))

    # preassign two lists to write the crosshair coordinates into
    list_of_centers = []
    list_of_centers2 = []
    x_coor = []
    n = 0
    # here we will extract the coordinates and radius of all the crosshairs in our image
    # we will also be drawing the circles onto the image to prove they have been detected
    for i in circles[0, :]:
        # draw the outer circle
        if i[0] > 60 and i[0] < 500 and i[1] > 100 and i[1] < 800:
            #cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            x_center, y_center = i[0], i[1]  # extract coordinates of center of circle
            x_coor.append(i[0])
            crosshair_center = tuple([x_center, y_center])  # create tuple of xy coordinates

            if n == 0:
                list_of_centers.append(crosshair_center)
                # this is our algorithm to attain how many pixels on our image span the width of a real world inch
                pix_per_inch = i[2] * 2 / 4
                radius = i[2]  # extract radius
                # cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)

            elif n >= 1:  # and x_coor[n - 1] < x_coor[n] - 50 or x_coor[n - 1] > x_coor[
                # n] + 50:  # filters our similar coordinates

                list_of_centers.append(crosshair_center)
                list_of_centers2.append(crosshair_center)
                # cv2.circle(image, crosshair_center, 35, (0, 255, 255), 2)
            n = n + 1
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # draw circles on crosshairs
    # print(circles)
    i=1
    for xy in list_of_centers:
        cv2.circle(cimg, xy, radius, (0, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cimg, str(i), xy, font, 1, (200, 255, 155), 2, cv2.LINE_AA)
        i = i + 1

    cv2.imshow('Pick a Target: ', cimg)
    cv2.waitKey(0)
    target = input('Which target will you be shooting at?: ')
    target = int(target) - 1  # turns user input string into integer
    settings.x_crosshair = list_of_centers[target][0]  # assigns variable to x coordinate of chosen target
    settings.y_crosshair = list_of_centers[target][1]  # assigns variable to y coordinate of chosen target
   #cv2.imshow('detected crosshairs', cimg)  # show the detected crosshairs
   # cv2.waitKey(0)

    return list_of_centers, pix_per_inch, radius, target




# /// This function finds similarities in two pictures and aligns those similarities in order to align the whole images//////
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
    #cv2.imwrite("matches.jpg", imMatches) #outputs image of matches

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




# //////////////// This function resizes our images to the same width and height (in pixels) to be able to accurately
# ////////////////////// csubtract them
def resizeImages(IMA, IMB):
    # find size of images
    heightA, widthA, channelA = IMA.shape

    # resize image 1 to image with shape [560,1020] for best results
    IMA = cv2.resize(IMA, (560, 1020), interpolation=cv2.INTER_NEAREST)

    # resize image2 to [560,1020] for best results
    IMB = cv2.resize(IMB, (560, 1020), interpolation=cv2.INTER_NEAREST)

    return IMA, IMB




# ////////////////////////////////Program Begins HERE/////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# This code will take two input images. The first image will be our target before a shot has been taken, the second image will be the
# the target AFTER the shot has been taken


#again = 1  # preassign variable again value of 1. Again defines whether the user will want to input more images (take
# more shots) 1=yes, 0 = np
#first = 1  # preassign value of 1 to first (this tells the code that it will be the first time running through an
# iteration of the whole code)

#while again == 1:  # The whole code will keep running while again =1
 #   if first == 1:  # This if statement will only be entered once in the beginning

        # construct the argument parse and parse the arguments
        # to run this program in terminal: python find_shot2.py --first image1name --second image2name
  #      ap = argparse.ArgumentParser()
   #     ap.add_argument("-f", "--first", required=True,  # takes as input our first image name
                        #help="first input image")
    #    ap.add_argument("-s", "--second", required=True,  # takes as input our second image name
                       # help="second")
     #   args = vars(ap.parse_args())

def calcElevationSecondary(y_inches, scope_MOA):
    # create list for elevation for continuing sessions
    e2 = []
    # if distance for y_inches is less than 0, then adjust elevation up
    if y_inches < 0:
        # append to list
        e2.append('Adjust elevation control ' + str(abs(round(y_inches/scope_MOA))) + ' clicks up')
    # if distance for y_inches is greater than 0, then adjust elevation down
    else:
        # append to list
        e2.append('Adjust elevation control ' + str(abs(round(y_inches/scope_MOA))) + ' clicks down')
    # return the number of clicks from first index of list
    return e2[0]

# function to calcuation windage for continuing sessions
# formula for windage = x_inches/(minute of angle)
def calcWindageSecondary(x_inches, scope_MOA):
    # create list for windage for continuing sessions
    w2 = []
    # if distance for x_inches is less than 0, then adjust windage right
    if x_inches < 0:
        # append to list
        w2.append('Adjust windage control ' + str(abs(round(x_inches/scope_MOA))) + ' clicks right')
    # if distance for x_inches is greater than 0, then adjust windage left
    else:
        # append to list
        w2.append('Adjust windage control ' + str(abs(round(x_inches/scope_MOA))) + ' clicks left')
    # return the number of clicks from first index of list
    return w2[0]


def processimage(imageA, imageB):
    distance_to_target_center_list = []  # assign variable as a list

    imageB, h = alignImages(imageB, imageA)  # this will enter our alignImages function
    imageA, imageB = resizeImages(imageA, imageB)  # this will enter our resizeImages function

    # We will align our images multiple times for more accurate aligning and results

    times = 5  # how many iterations the alignImages function will have
    i=0
    while i < times:
        imageB, h = alignImages(imageB, imageA)
        i += 1
    copy_A = imageA.copy()
    copy_B = imageB.copy()

    # these 5 lines of code will draw circles around our crosshairs and label them for the user
    # to be able to pick which one he wants to shoot at
    #i = 1
    #for xy in settings.list_of_centers:
     #   cv2.circle(imageC, xy, settings.radius, (0, 255, 255), 2)
      #  font = cv2.FONT_HERSHEY_SIMPLEX
       # cv2.putText(imageC, str(i), xy, font, 1, (200, 255, 155), 2, cv2.LINE_AA)
        #i = i + 1

    # Will be asking the user to pick which target he will be shooting at
    #cv2.imshow('Pick a Target: ', imageC)
    #cv2.waitKey(0)
    #target = input('Which target will you be shooting at?: ')
    #target = int(target) - 1  # turns user input string into integer
    #settings.x_crosshair = settings.list_of_centers[target][0]  # assigns variable to x coordinate of chosen target
    #settings.y_crosshair = settings.list_of_centers[target][1]  # assigns variable to y coordinate of chosen target

    # ///////////here we will begin our image subtraction algorithm///////////
    # this will find the shot hole

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale',grayA)
    #cv2.waitKey(0)
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # find avg area of found differences in order to use for filtering effects
    areas = []

    # iterate through all the contours (differences)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        wa, ha, c = imageA.shape  # assign wa and ha to the shape of our image
        picture_area = wa * ha  # size of our whole image
        area = w * h  # size/area of the current contour
        if area < picture_area / 4:  # this filters out differences that have very large areas
            areas.append(area)  # append the filtered areas into list

    # loop over the contours
    i = 0
    avg_area = sum(areas) / (len(areas) + 1)  # we take the average area of our remaining differences

    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)  # extract width and height of contour

        # we need to filter out some more differences
        area = w * h  # area of current contour
        heightA, widthA, channelA = imageA.shape

        # conditional statement to ignore contours outside a radius of 4 inches from our chosen target
        if abs(x - settings.x_crosshair) * settings.pix_per_inch ** -1 < 4 and abs(
                settings.y_crosshair - y) * settings.pix_per_inch ** -1 < 4:

            # filter our contours with micro areas and contours with too large areas
            if area > avg_area / 2 and area < picture_area / 10:

                # filter out irregular shapes
                if w < h * 1.4 and w > h * .6:

                    # ignore the edges of our images as our targets will be located throughout the center
                    if x > 0 + widthA / 8 and x < widthA - widthA / 8 and y > 0 + heightA / 8 and y < heightA - heightA / 8:
                        i = i + 1
                        # if the contour got through the filters, will will draw a circle around it
                        #cv2.circle(imageA, (int(x + w / 2), int(y + h / 2)), 3, (0, 0, 255), 2)
                        cv2.circle(imageB, (int(x + w / 2), int(y + h / 2)), 3, (0, 0, 255), 2)

                        shot_coor_x = x  # assign variable to x coordinate of shot locations
                        shot_coor_y = y  # assign variable to y coordinate of shot location

                        # calculate the horizontal and vertical distance between our shot and the target center
                        horizontal_diff = (x + w / 2 - settings.x_crosshair) * settings.pix_per_inch ** -1  # positive means right
                        vertical_diff = (settings.y_crosshair - y - h / 2) * settings.pix_per_inch ** -1  # positive means high

                        # calculate total distance from shot location to target location in inches
                        distance_to_target_center = (horizontal_diff ** 2 + vertical_diff ** 2) ** (
                                1 / 2)

                        distance_to_target_center_list.append(distance_to_target_center)  # create  list of distances

                        # draw a line from the center of the crosshair to the shot location
                        cv2.line(imageB, (int(x + w / 2), int(y + h / 2)), (settings.list_of_centers[settings.target]), (0, 255, 0), 2)
                        # output distances from hole to target to the user
                        if horizontal_diff <= 0:
                            print('Hit too left by: ' + str(abs(horizontal_diff)) + ' in')
                        elif horizontal_diff > 0:
                            print('Hit too right by: ' + str(horizontal_diff) + ' in')
                        if vertical_diff > 0:
                            print('Hit to high by: ' + str(vertical_diff) + ' in')
                        elif vertical_diff < 0:
                            print('Hit too low by: ' + str(abs(vertical_diff)) + ' in')

                        print('distance from center of target (in) ' + str(distance_to_target_center))
    e2 = calcElevationSecondary(vertical_diff, .25)
    w2 = calcWindageSecondary(horizontal_diff, .25)
    print(e2)
    print(w2)
    i = str(i)
    # draw a circle around the chosen target
    for xy in settings.list_of_centers:
        cv2.circle(imageB, (settings.x_crosshair, settings.y_crosshair), settings.radius, (0, 255, 255), 2)
    # print('number of matches: ' + i)
    # show the output images

    # calculate average distance from target to hole from all shots taken so far and standard deviation
    avg_distance = sum(distance_to_target_center_list) / len(distance_to_target_center_list)
    if len(distance_to_target_center_list) != 1:
        std_dev = statistics.stdev(distance_to_target_center_list)
        print('The average distance from the center of the target is: ' + str(
            avg_distance) + ' with a standard deviation of: ' + str(std_dev))
    # show images with spotted hole
    cv2.imshow("Before", imageA)
    cv2.imshow("After", copy_B)
    cv2.waitKey(0)
    cv2.imshow("Hit Found!", imageB)
    #cv2.imshow("Diff", diff)
    #cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)
    # ask the user if he will be shooting again
    #again = input('WIll you shoot again yes(1), no (0): ')
    #again = int(again)  # assign variable again to either 1 or 2
   # first = 0;  # change value of variable 'first'
    #last_image = copy_B.copy();  # make a copy of clean version of second image to become first
    # in next iteration
    return copy_A, copy_B
