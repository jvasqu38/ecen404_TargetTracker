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
from watcher import begin_watching


# Actual Kivy layout begins here

class StartWindow(Screen):
    def pressed(self):
        begin_watching()
        #src_path = sys.argv[1] if len(sys.argv) > 1 else '.' #runs watcher
        #ImagesWatcher(src_path).run() #runs watcher



class MainWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class MyApp(App):
    def build(self):
        return kv

#def begin_watching(src_path):
  #  return




if __name__ == "__main__":
    #src_path = sys.argv[1] if len(sys.argv) > 1 else '.'  # these two were indented
    #print(src_path)
    MyApp().run()





