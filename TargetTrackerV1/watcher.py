import os
import settings
import cv2
from find_shot2 import processimage
from watchdog.events import RegexMatchingEventHandler
import sys
import time
from watchdog.observers import Observer
settings.init()

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
        filename = f"{filename}.jpg"

        imageB = cv2.imread(filename)



        if settings.Iterations == 1:
            settings.imageA = cv2.imread(filename)
            cv2.imshow('original', settings.imageA)
            cv2.waitKey(0)
            print('i=1')
            settings.Iterations = settings.Iterations + 1

        else:
            print('i=/1')
            processimage(settings.imageA, imageB)
            #cv2.imshow('A', settings.imageA)
            #cv2.imshow('B', imageB)
            #cv2.waitKey(0)
            settings.imageA = imageB
            settings.Iterations = settings.Iterations + 1







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
    src_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    ImagesWatcher(src_path).run()