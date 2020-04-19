import os
import settings
import cv2
from find_shot2 import processimage
from find_shot2 import find_crosshairs
from watchdog.events import RegexMatchingEventHandler
import sys
import time
from watchdog.observers import Observer
#from main import show_popup





class ImagesEventHandler(RegexMatchingEventHandler):
    THUMBNAIL_SIZE = (128, 128)
    IMAGES_REGEX = [r".*[^_thumbnail]\.jpg$"]

    def __init__(self):
        super().__init__(self.IMAGES_REGEX)

    def on_created(self, event):
        file_size = -1
        while file_size != os.path.getsize(event.src_path):
            file_size = os.path.getsize(event.src_path)
            time.sleep(1)
        self.process(event)

    def process(self, event):
        filename, ext = os.path.splitext(event.src_path)
        settings.filename = f"{filename}.jpg"

        settings.imageB = cv2.imread(settings.filename)



        if settings.Iterations == 1:
            settings.imageA = cv2.imread(settings.filename)
            #need to scale image and align first
            #settings.list_of_centers, settings.pix_per_inch, settings.radius = find_crosshairs(settings.imageA)
            settings.list_of_centers, settings.pix_per_inch, settings.radius, settings.target = find_crosshairs(settings.imageA)
            #show_popup()
            #find_coordinates(settings.target)

            print('i=1')
            settings.Iterations = settings.Iterations + 1

        else:
            print('i=/1')
            settings.imageA, imageB = processimage(settings.imageA, settings.imageB)
            #cv2.imshow('A', settings.imageA)
            #cv2.imshow('B', imageB)
            #cv2.waitKey(0)
            settings.imageA = settings.imageB
            settings.list_of_centers, settings.pix_per_inch, settings.radius, settings.target = find_crosshairs(settings.imageA)
            settings.Iterations = settings.Iterations + 1
        print('Waiting for you to shoot...')







class ImagesWatcher:
    def __init__(self, src_path):
        self.__src_path = src_path
        self.__event_handler = ImagesEventHandler()
        self.__event_observer = Observer()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def start(self):
        self.__schedule()
        self.__event_observer.start()

    def stop(self):
        self.__event_observer.stop()
        self.__event_observer.join()

    def __schedule(self):
        self.__event_observer.schedule(
            self.__event_handler,
            self.__src_path,
            recursive=True
        )



if __name__ == "__main__":
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'  # these two were indented
    settings.init()
    print("Take a picture of your target")
    ImagesWatcher(src_path).run()
