import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from watcher import begin_watching
import multiprocessing
import os
import settings
import cv2
from find_shot2 import processimage
from find_shot2 import find_crosshairs
from watchdog.events import RegexMatchingEventHandler
import sys
import time
from watchdog.observers import Observer

# Actual Kivy layout begins here
p1 = multiprocessing.Process(target=begin_watching)
lock = multiprocessing.Lock()

#def run_gui():
#    MyApp().run()
 #   return

class StartWindow(Screen):
    def pressed(self):
        #p1 = multiprocessing.Process(target=begin_watching)
        p1.start()
        return
        #src_path = sys.argv[1] if len(sys.argv) > 1 else '.' #runs watcher
        #ImagesWatcher(src_path).run() #runs watcher



class MainWindow(Screen):
    img_path = StringProperty(None)
    target_num = ObjectProperty(None)

    def pressed_main(self):
        lock.acquire()
        self.img_path = settings.filename
        targets = self.target_num.text
        settings.target = int(targets) - 1
        print(settings.target)
        lock.release()
    pass

#class Pick_Target(FloatLayout):
 #   pass

def show_popup():
    show = Pick_Target()
    popupWindow = Popup(title = "Pick Target", content = show, size_hint = (None,None), size=(400,400))
    popupWindow.open()

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class MyApp(App):
    def build(self):
        return kv


#settings.init()
if __name__ == "__main__":
    MyApp().run()
    #p2 = multiprocessing.Process(target = run_gui)
    #p2.start()


    # if __name__ == "__main__":
    #ImagesWatcher(src_path).run()





