import kivy
kivy.require('1.0.7')

from kivy.config import Config
Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '800')
Config.set('graphics', 'resizable', False)

from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.switch import Switch
from kivy.graphics import Color, Rectangle

from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scatter import Scatter
from kivy.uix.textinput import TextInput

from kivy.core.window import Window
Window.clearcolor = (0.8, 0.8, 0.8, 1)

# dimensions are set for a 7 inch tablet

class TestApp(App):

    def build(self):
        layout = FloatLayout()

        # return a Button() as a root widget
        start = ToggleButton(text="START",
                       font_size="40sp",
                       bold="True",
                       background_color=(0.32, 0.66, 0.12, 1),
                       background_normal='',
                       color=(1,1,1,1),
                       size=(1,1),
                       size_hint=(.12,.12),
                       pos=(20,20))

        end = ToggleButton(text="END",
                       font_size="40sp",
                       bold="True",
                       background_color=(0.85, 0.26, 0.24, 1),
                       background_normal='',
                       color=(1, 1, 1, 1),
                       size=(1, 1),
                       size_hint=(.12, .12),
                       pos=(1105, 20))

        layout.add_widget(start)
        layout.add_widget(end)

        align_scope = Label(text='Align\nScope?', font_size='40sp', pos=(510,20), size=(180,120), size_hint=(0.12, 0.12),
                            color=(0,0,0,1), bold=True)

        layout.add_widget(align_scope)

        switch = Switch(active=False, pos=(650,20), size_hint=(.12, .12), size=(1,1))
        layout.add_widget(switch)

        choose_target = ToggleButton(text="Choose\n Target",
                       font_size="35sp",
                       bold="True",
                       background_color=(0.31, 0.4, 0.65, 1),
                       background_normal='',
                       color=(1,1,1,1),
                       size=(1,1),
                       size_hint=(.12,.12),
                       pos=(280,20))

        change_target = ToggleButton(text="Change\n Target?",
                                     font_size="35sp",
                                     bold="True",
                                     background_color=(0.72, 0.24, 0.58, 1),
                                     background_normal='',
                                     color=(1, 1, 1, 1),
                                     size=(1, 1),
                                     size_hint=(.12, .12),
                                     pos=(850, 20))

        layout.add_widget(choose_target)
        layout.add_widget(change_target)

        target = Image(source='photo.jpg', size=(200,20), pos=(-280,50))

        layout.add_widget(target)

        return layout




if __name__ == '__main__':
    TestApp().run()