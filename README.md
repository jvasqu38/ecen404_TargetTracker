# ecen404_TargetTracker
Target Tracker System for senior design.

This project's purpose is to keep track of the bulletholes on a target. 
The file watcher.py monitors the directory in which it is in for new files. The purpose of watcher.py is to wait for new images
of the target and activate the image processing code using the new images as input.

find_shot2.py takes as input two images. First image should be a picture of the target previous to have been shot at. 
The second image should be the target with a new bullethole. This code then finds where the bullet hit and extracts data from it.
