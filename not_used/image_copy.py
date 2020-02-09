import cv2
import settings
import watcher
def ProcessImage(image_path,Iterations):
    image_path = str(image_path)
    imageB = cv2.imread(image_path)


    if Iterations == 1:
        settings.imageA = imageB.copy()
        return
    else:
        cv2.imshow('A', settings.imageA)
        cv2.imshow('B', imageB)
        cv2.waitKey(0)
        settings.imageA = imageB.copy()
        Iterations = Iterations+1

    #else:
     #   imageA = cv2.imread(image_path)
      #  cv2.imshow('A',imageA)
       # cv2.imshow('B',imageB)
        #cv2.waitKey(0)


    #return
