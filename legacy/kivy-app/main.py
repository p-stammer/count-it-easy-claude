from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRectangleFlatButton, MDIconButton, MDFlatButton
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog

from kivy.core.window import Window
from kivy.uix.modalview import ModalView
from kivy.uix.image import Image
from kivy.clock import Clock, mainthread
from kivy.uix.button import Button
from kivy.base import EventLoop
from kivy import platform
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.metrics import dp

from camera4kivy import Preview

import numpy as np
# import os
# os.chdir(os.path.abspath(os.path.dirname(__file__)))

try:
    # Import TFLite interpreter from tflite_runtime package if it's available.
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # If not, fallback to use the TFLite interpreter from the full TF package.
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

if platform == "android":
    from android import mActivity
    from android.permissions import request_permissions, check_permission, Permission


class AndroidPermissions:
    def __init__(self, start_app=None):
        self.permission_dialog_count = 0
        self.start_app = start_app
        if platform == "android":
            self.permissions = [Permission.CAMERA]

            self.permission_status([], [])
        elif self.start_app:
            self.start_app()

    def permission_status(self, permissions, grants):
        granted = True
        for p in self.permissions:
            granted = granted and check_permission(p)
        if granted:
            if self.start_app:
                self.start_app()
        elif self.permission_dialog_count < 2:
            Clock.schedule_once(self.permission_dialog)
        else:
            self.no_permission_view()

    def permission_dialog(self, dt):
        self.permission_dialog_count += 1
        request_permissions(self.permissions, self.permission_status)

    @mainthread
    def no_permission_view(self):
        view = ModalView()
        view.add_widget(
            Button(
                text="Permission NOT granted.\n\n"
                + "Tap to quit app.\n\n\n"
                + 'If you selected "Don\'t Allow",\n'
                + "enable permission with App Settings.",
                on_press=self.bye,
            )
        )
        view.open()

    def bye(self, instance):
        mActivity.finishAndRemoveTask()


class CustomModel:
    def __init__(self, path):
        self._load_model(path)

    def _load_model(self, path):
        self.interpreter = Interpreter(model_path=path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()[0]["index"]

    def predict(self, im):
        self.interpreter.tensor(self.input_details)()[0][:, :] = im
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details)
        return out


class CustomPreview(Preview):
    def analyze_pixels_callback(
        self, pixels, image_size, image_pos, image_scale, mirror
    ):
        rgba = np.frombuffer(pixels, np.uint8).reshape(image_size[1], image_size[0], 4)
        self.current_image = rgba[:, :, :3]
        self.im_shape = self.current_image.shape[:2]


class CountItEasy(MDApp):
    def __init__(self, **kwargs):
        super(CountItEasy, self).__init__(**kwargs)
        Window.bind(on_keyboard=self.Android_back_click)

    def Android_back_click(self, window, key, *largs):
        if key == 27 or key == 1001:
            self.on_click_back_button()
            return True

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"

        self.active_bo3_session = False
        self.inside_bo3_screens = False
        self.active_bo3_profile_indices = []
        self.forward_to_counting = False
        self.go_to_bo3_stats_table = False
        self.counting_colors = [
            "#88B358",
            "#BA2525",
            "#FFEE00",
            "#007599",
            "#ffddfb",
            "#82ffff",
            "#dddddd",
            "#fd9b1c",
            "#ff00e6",
        ]
        self.rdm_color_idx = [0, 0, 0]
        self.point_string = ""

        sm = ScreenManager()
        screen_home = Screen(name="Home")
        screen_profile = Screen(name="Profiles")
        screen_choose_single_profile = Screen(name="Choose_single_profile")
        screen_choose_bo3_profiles = Screen(name="Choose_bo3_profiles")
        screen_choose_bo3_profile_to_count = Screen(name="Choose_bo3_profile_to_count")
        screen_camera = Screen(name="Camera")
        screen_counting = Screen(name="Counting")
        screen_stats = Screen(name="Stats")
        screen_bo3_stats = Screen(name="Bo3_stats")
        screen_info = Screen(name="Info")
        screen_settings = Screen(name="Settings")

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_home = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.9},
            halign="center",
            font_style="H3",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )

        count_it_button = MDRectangleFlatButton(
            text="Count it!", font_style="H6", line_width=1.2, text_color = (1,1,1,0.9)
        )
        count_it_button.bind(on_press=self.on_click_count_it_button)
        count_it_button.size_hint = (0.5, 0.1)
        count_it_button.pos_hint = {"center_x": 0.5, "center_y": 0.7}
        count_it_button.line_color = "#FFEE00"

        bo3_button = MDRectangleFlatButton(
            text="Best of 3", font_style="H6", line_width=1.2, text_color = (1,1,1,0.9)
        )
        bo3_button.bind(on_press=self.on_click_bo3_button)
        bo3_button.size_hint = (0.5, 0.1)
        bo3_button.pos_hint = {"center_x": 0.5, "center_y": 0.55}
        bo3_button.line_color = "#88B358"
        profiles_button = MDRectangleFlatButton(
            text="Profiles", font_style="H6", line_width=1.2, text_color = (1,1,1,0.9)
        )
        profiles_button.bind(on_press=self.on_click_profiles_button)
        profiles_button.size_hint = (0.5, 0.1)
        profiles_button.pos_hint = {"center_x": 0.5, "center_y": 0.4}
        profiles_button.line_color = "#fd9b1c"
        stats_button = MDRectangleFlatButton(
            text="Stats", font_style="H6", line_width=1.2, text_color = (1,1,1,0.9)
        )
        stats_button.bind(on_press=self.on_click_stats_button)
        stats_button.size_hint = (0.5, 0.1)
        stats_button.pos_hint = {"center_x": 0.5, "center_y": 0.25}
        stats_button.line_color = "#ff00e6"
        info_button = MDIconButton(icon="information")
        info_button.bind(on_press=self.on_click_info_button)
        # info_button.size_hint = (0.05, 0.05)
        info_button.pos_hint = {"center_x": 0.9, "center_y": 0.05}

        settings_button = MDIconButton(icon="cog")
        settings_button.bind(on_press=self.on_click_settings_button)
        # info_button.size_hint = (0.05, 0.05)
        settings_button.pos_hint = {"center_x": 0.1, "center_y": 0.05}

        bl_home.add_widget(cie_logo)
        bl_home.add_widget(count_it_button)
        bl_home.add_widget(bo3_button)
        bl_home.add_widget(profiles_button)
        bl_home.add_widget(stats_button)
        bl_home.add_widget(info_button)
        bl_home.add_widget(settings_button)

        self.names = [
            name[1:] for name in open("highscores.txt").readline()[1:-1].split(",")
        ]
        self.highscore_values = (
            np.loadtxt("highscores.txt", delimiter=",").astype(np.int32).tolist()
        )
        self.stats = np.loadtxt("stats.txt", delimiter=",")
        self.history_0 = np.loadtxt("history_0.txt", delimiter=",")
        self.history_1 = np.loadtxt("history_1.txt", delimiter=",")
        self.history_2 = np.loadtxt("history_2.txt", delimiter=",")
        self.history_3 = np.loadtxt("history_3.txt", delimiter=",")
        self.history_4 = np.loadtxt("history_4.txt", delimiter=",")
        self.history_5 = np.loadtxt("history_5.txt", delimiter=",")

        self.bo3_round = 0
        self.bo3_session_scores = np.loadtxt("bo3_session_scores.txt", delimiter=",")
        if not (self.bo3_session_scores == 0).all():
            self.active_bo3_profile_indices = np.nonzero(
                np.loadtxt("bo3_session_indices.txt", delimiter=",")
            )[0].tolist()
            self.active_bo3_session = True
            self.bo3_round = np.nonzero(self.bo3_session_scores)[1].max()
            if (
                self.bo3_session_scores[self.active_bo3_profile_indices, self.bo3_round]
                != 0
            ).all() and (not self.bo3_round == 2):
                self.bo3_round += 1

        self.history = [
            self.history_0,
            self.history_1,
            self.history_2,
            self.history_3,
            self.history_4,
            self.history_5,
        ]
        for i, h in enumerate(self.history):
            if len(h.shape) < 1:
                self.history[i] = h[..., None]
            self.history[i] = self.history[i].tolist()

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_profile = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        profile_button_0 = MDRectangleFlatButton(
            text=self.names[0] if not "New0" in self.names[0] else "Add Profile",
            font_style="H6",
            line_width=1.2, text_color = (1,1,1,0.9)
        )
        profile_button_0.size_hint = (0.6, 0.1)
        profile_button_0.pos_hint = {"center_x": 0.5, "center_y": 0.8}
        profile_button_0.bind(on_press=self.on_click_profile_button_0)
        profile_button_0.line_color = "#007599"
        profile_button_1 = MDRectangleFlatButton(
            text=self.names[1] if not "New1" in self.names[1] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_1.size_hint = (0.6, 0.1)
        profile_button_1.pos_hint = {"center_x": 0.5, "center_y": 0.675}
        profile_button_1.bind(on_press=self.on_click_profile_button_1)
        profile_button_1.line_color = "#BA2525"
        profile_button_2 = MDRectangleFlatButton(
            text=self.names[2] if not "New2" in self.names[2] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_2.size_hint = (0.6, 0.1)
        profile_button_2.pos_hint = {"center_x": 0.5, "center_y": 0.55}
        profile_button_2.bind(on_press=self.on_click_profile_button_2)
        profile_button_2.line_color = "#ffddfb"
        profile_button_3 = MDRectangleFlatButton(
            text=self.names[3] if not "New3" in self.names[3] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_3.line_color = "#FFEE00"
        profile_button_3.size_hint = (0.6, 0.1)
        profile_button_3.pos_hint = {"center_x": 0.5, "center_y": 0.425}
        profile_button_3.bind(on_press=self.on_click_profile_button_3)
        profile_button_4 = MDRectangleFlatButton(
            text=self.names[4] if not "New4" in self.names[4] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_4.line_color = "#82ffff"
        profile_button_4.size_hint = (0.6, 0.1)
        profile_button_4.pos_hint = {"center_x": 0.5, "center_y": 0.3}
        profile_button_4.bind(on_press=self.on_click_profile_button_4)
        profile_button_5 = MDRectangleFlatButton(
            text=self.names[5] if not "New5" in self.names[5] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_5.line_color = "#fd9b1c"
        profile_button_5.size_hint = (0.6, 0.1)
        profile_button_5.pos_hint = {"center_x": 0.5, "center_y": 0.175}
        profile_button_5.bind(on_press=self.on_click_profile_button_5)
        delete_button_0 = MDIconButton(icon="delete")
        delete_button_0.size_hint = (0.1, 0.1)
        delete_button_0.pos_hint = {"center_x": 0.9, "center_y": 0.8}
        delete_button_0.bind(on_press=self.on_click_delete_button_0)
        delete_button_1 = MDIconButton(icon="delete")
        delete_button_1.size_hint = (0.1, 0.1)
        delete_button_1.pos_hint = {"center_x": 0.9, "center_y": 0.675}
        delete_button_1.bind(on_press=self.on_click_delete_button_1)
        delete_button_2 = MDIconButton(icon="delete")
        delete_button_2.size_hint = (0.1, 0.1)
        delete_button_2.pos_hint = {"center_x": 0.9, "center_y": 0.55}
        delete_button_2.bind(on_press=self.on_click_delete_button_2)
        delete_button_3 = MDIconButton(icon="delete")
        delete_button_3.size_hint = (0.1, 0.1)
        delete_button_3.pos_hint = {"center_x": 0.9, "center_y": 0.425}
        delete_button_3.bind(on_press=self.on_click_delete_button_3)
        delete_button_4 = MDIconButton(icon="delete")
        delete_button_4.size_hint = (0.1, 0.1)
        delete_button_4.pos_hint = {"center_x": 0.9, "center_y": 0.3}
        delete_button_4.bind(on_press=self.on_click_delete_button_4)
        delete_button_5 = MDIconButton(icon="delete")
        delete_button_5.size_hint = (0.1, 0.1)
        delete_button_5.pos_hint = {"center_x": 0.9, "center_y": 0.175}
        delete_button_5.bind(on_press=self.on_click_delete_button_5)
        confirm_delete_button = MDFlatButton(
            text="DELETE",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        confirm_delete_button.bind(on_press=self.on_click_confirm_delete_button)
        cancel_delete_button = MDFlatButton(
            text="CANCEL",
        )
        cancel_delete_button.bind(on_press=self.on_click_cancel_delete_button)

        self.confirm_delete_popup = MDDialog(
            title="Delete Profile",
            type="custom",
            buttons=[
                cancel_delete_button,
                confirm_delete_button,
            ],
            size_hint=(0.6, 0.3),
        )

        new_profile_popup_content = MDFloatLayout()
        new_profile_popup_input = MDTextField(
            hint_text="Enter your name", pos_hint={"center_x": 0.5, "center_y": 0.5}
        )
        new_profile_popup_confirm_button = MDFlatButton(
            text="OK",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        new_profile_popup_confirm_button.bind(
            on_press=self.on_click_confirm_new_profile_button
        )
        new_profile_popup_cancel_button = MDFlatButton(
            text="CANCEL",
        )
        new_profile_popup_cancel_button.bind(
            on_press=self.on_click_cancel_new_profile_button
        )
        new_profile_popup_content.add_widget(new_profile_popup_input)
        new_profile_popup = MDDialog(
            title="Add new profile:",
            type="custom",
            content_cls=new_profile_popup_content,
            buttons=[
                new_profile_popup_cancel_button,
                new_profile_popup_confirm_button,
            ],
        )

        new_profile_popup.size_hint = (0.6, 0.3)
        new_profile_popup.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        back_button_profile = MDIconButton(icon="reply")
        back_button_profile.bind(on_press=self.on_click_back_button)
        # back_button_profile.size_hint = (0.05, 0.05)
        back_button_profile.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button_profile.background_color = self.theme_cls.primary_color
        # back_button_profile.color = (1, 1, 1, 1)

        bl_profile.add_widget(cie_logo)
        bl_profile.add_widget(back_button_profile)
        bl_profile.add_widget(profile_button_0)
        bl_profile.add_widget(profile_button_1)
        bl_profile.add_widget(profile_button_2)
        bl_profile.add_widget(profile_button_3)
        bl_profile.add_widget(profile_button_4)
        bl_profile.add_widget(profile_button_5)
        # bl_profile.add_widget(new_profile_popup)
        bl_profile.add_widget(delete_button_0)
        bl_profile.add_widget(delete_button_1)
        bl_profile.add_widget(delete_button_2)
        bl_profile.add_widget(delete_button_3)
        bl_profile.add_widget(delete_button_4)
        bl_profile.add_widget(delete_button_5)
        sm.ids["profiles_profile_button_0"] = profile_button_0
        sm.ids["profiles_profile_button_1"] = profile_button_1
        sm.ids["profiles_profile_button_2"] = profile_button_2
        sm.ids["profiles_profile_button_3"] = profile_button_3
        sm.ids["profiles_profile_button_4"] = profile_button_4
        sm.ids["profiles_profile_button_5"] = profile_button_5
        sm.ids["profiles_new_profile_input"] = new_profile_popup_input
        sm.ids["profiles_new_profile_popup"] = new_profile_popup
        sm.ids["profiles_delete_button_0"] = delete_button_0
        sm.ids["profiles_delete_button_1"] = delete_button_1
        sm.ids["profiles_delete_button_2"] = delete_button_2
        sm.ids["profiles_delete_button_3"] = delete_button_3
        sm.ids["profiles_delete_button_4"] = delete_button_4
        sm.ids["profiles_delete_button_5"] = delete_button_5

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_choose_bo3_profile_to_count = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        back_button = MDIconButton(icon="reply")
        back_button.bind(on_press=self.on_click_back_button)
        # back_button.size_hint = (0.05, 0.05)
        back_button.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button.background_color = self.theme_cls.primary_color
        # back_button.color = (1, 1, 1, 1)
        profile_button_0 = MDRectangleFlatButton(
            text=self.names[0] if not "New0" in self.names[0] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_0.size_hint = (0.6, 0.1)
        profile_button_0.pos_hint = {"center_x": 0.5, "center_y": 0.8}
        profile_button_0.bind(on_press=self.on_click_profile_button_0)
        profile_button_0.line_color = "#007599"
        profile_button_1 = MDRectangleFlatButton(
            text=self.names[1] if not "New1" in self.names[1] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_1.size_hint = (0.6, 0.1)
        profile_button_1.pos_hint = {"center_x": 0.5, "center_y": 0.675}
        profile_button_1.bind(on_press=self.on_click_profile_button_1)
        profile_button_1.line_color = "#BA2525"
        profile_button_2 = MDRectangleFlatButton(
            text=self.names[2] if not "New2" in self.names[2] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_2.size_hint = (0.6, 0.1)
        profile_button_2.pos_hint = {"center_x": 0.5, "center_y": 0.55}
        profile_button_2.bind(on_press=self.on_click_profile_button_2)
        profile_button_2.line_color = "#ffddfb"
        profile_button_3 = MDRectangleFlatButton(
            text=self.names[3] if not "New3" in self.names[3] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_3.line_color = "#FFEE00"
        profile_button_3.size_hint = (0.6, 0.1)
        profile_button_3.pos_hint = {"center_x": 0.5, "center_y": 0.425}
        profile_button_3.bind(on_press=self.on_click_profile_button_3)
        profile_button_4 = MDRectangleFlatButton(
            text=self.names[4] if not "New4" in self.names[4] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_4.line_color = "#82ffff"
        profile_button_4.size_hint = (0.6, 0.1)
        profile_button_4.pos_hint = {"center_x": 0.5, "center_y": 0.3}
        profile_button_4.bind(on_press=self.on_click_profile_button_4)
        profile_button_5 = MDRectangleFlatButton(
            text=self.names[5] if not "New5" in self.names[5] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_5.line_color = "#fd9b1c"
        profile_button_5.size_hint = (0.6, 0.1)
        profile_button_5.pos_hint = {"center_x": 0.5, "center_y": 0.175}
        profile_button_5.bind(on_press=self.on_click_profile_button_5)
        self.round_label_color_prefix = "#ff00e6"
        bo3_round_label = MDLabel(
            text=f"[color={self.round_label_color_prefix}]Round {self.bo3_round + 1}[/color]",
            halign="center",
            valign="center",
            markup=True,
            pos_hint={"center_x": 0.9, "center_y": 0.95},
        )
        bo3_stats_button = MDRectangleFlatButton(
            text="Standings",
            font_style="H6",
            size_hint=(0.6, 0.05),
            pos_hint={"center_x": 0.5, "center_y": 0.05},
            line_color="#dddddd",
            line_width=1.2,
        )
        bo3_stats_button.bind(on_press=self.on_click_bo3_stats_button)
        bo3_stats_button.background_color = self.theme_cls.primary_color
        bo3_stats_button.color = (1, 1, 1, 1)

        bl_choose_bo3_profile_to_count.add_widget(cie_logo)
        bl_choose_bo3_profile_to_count.add_widget(bo3_stats_button)
        bl_choose_bo3_profile_to_count.add_widget(bo3_round_label)
        bl_choose_bo3_profile_to_count.add_widget(back_button)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_0)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_1)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_2)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_3)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_4)
        bl_choose_bo3_profile_to_count.add_widget(profile_button_5)

        sm.ids["bo3_round_label"] = bo3_round_label
        sm.ids["bo3_profile_button_0"] = profile_button_0
        sm.ids["bo3_profile_button_1"] = profile_button_1
        sm.ids["bo3_profile_button_2"] = profile_button_2
        sm.ids["bo3_profile_button_3"] = profile_button_3
        sm.ids["bo3_profile_button_4"] = profile_button_4
        sm.ids["bo3_profile_button_5"] = profile_button_5

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_choose_single_profile = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        back_button0 = MDIconButton(icon="reply")
        back_button0.bind(on_press=self.on_click_back_button)
        # back_button0.size_hint = (0.05, 0.05)
        back_button0.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button0.background_color = self.theme_cls.primary_color
        # back_button0.color = (1, 1, 1, 1)
        profile_button_0 = MDRectangleFlatButton(
            text=self.names[0] if not "New0" in self.names[0] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_0.size_hint = (0.6, 0.1)
        profile_button_0.pos_hint = {"center_x": 0.5, "center_y": 0.8}
        profile_button_0.bind(on_press=self.on_click_profile_button_0)
        profile_button_0.line_color = "#007599"
        profile_button_1 = MDRectangleFlatButton(
            text=self.names[1] if not "New1" in self.names[1] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_1.size_hint = (0.6, 0.1)
        profile_button_1.pos_hint = {"center_x": 0.5, "center_y": 0.675}
        profile_button_1.bind(on_press=self.on_click_profile_button_1)
        profile_button_1.line_color = "#BA2525"
        profile_button_2 = MDRectangleFlatButton(
            text=self.names[2] if not "New2" in self.names[2] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_2.size_hint = (0.6, 0.1)
        profile_button_2.pos_hint = {"center_x": 0.5, "center_y": 0.55}
        profile_button_2.bind(on_press=self.on_click_profile_button_2)
        profile_button_2.line_color = "#ffddfb"
        profile_button_3 = MDRectangleFlatButton(
            text=self.names[3] if not "New3" in self.names[3] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_3.line_color = "#FFEE00"
        profile_button_3.size_hint = (0.6, 0.1)
        profile_button_3.pos_hint = {"center_x": 0.5, "center_y": 0.425}
        profile_button_3.bind(on_press=self.on_click_profile_button_3)
        profile_button_4 = MDRectangleFlatButton(
            text=self.names[4] if not "New4" in self.names[4] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_4.line_color = "#82ffff"
        profile_button_4.size_hint = (0.6, 0.1)
        profile_button_4.pos_hint = {"center_x": 0.5, "center_y": 0.3}
        profile_button_4.bind(on_press=self.on_click_profile_button_4)
        profile_button_5 = MDRectangleFlatButton(
            text=self.names[5] if not "New5" in self.names[5] else "Add Profile",
            font_style="H6",
            line_width=1.2,text_color = (1,1,1,0.9)
        )
        profile_button_5.line_color = "#fd9b1c"
        profile_button_5.size_hint = (0.6, 0.1)
        profile_button_5.pos_hint = {"center_x": 0.5, "center_y": 0.175}
        profile_button_5.bind(on_press=self.on_click_profile_button_5)
        delete_button_0 = MDIconButton(icon="delete")
        delete_button_0.size_hint = (0.1, 0.1)
        delete_button_0.pos_hint = {"center_x": 0.9, "center_y": 0.8}
        delete_button_0.bind(on_press=self.on_click_delete_button_0)
        delete_button_1 = MDIconButton(icon="delete")
        delete_button_1.size_hint = (0.1, 0.1)
        delete_button_1.pos_hint = {"center_x": 0.9, "center_y": 0.675}
        delete_button_1.bind(on_press=self.on_click_delete_button_1)
        delete_button_2 = MDIconButton(icon="delete")
        delete_button_2.size_hint = (0.1, 0.1)
        delete_button_2.pos_hint = {"center_x": 0.9, "center_y": 0.55}
        delete_button_2.bind(on_press=self.on_click_delete_button_2)
        delete_button_3 = MDIconButton(icon="delete")
        delete_button_3.size_hint = (0.1, 0.1)
        delete_button_3.pos_hint = {"center_x": 0.9, "center_y": 0.425}
        delete_button_3.bind(on_press=self.on_click_delete_button_3)
        delete_button_4 = MDIconButton(icon="delete")
        delete_button_4.size_hint = (0.1, 0.1)
        delete_button_4.pos_hint = {"center_x": 0.9, "center_y": 0.3}
        delete_button_4.bind(on_press=self.on_click_delete_button_4)
        delete_button_5 = MDIconButton(icon="delete")
        delete_button_5.size_hint = (0.1, 0.1)
        delete_button_5.pos_hint = {"center_x": 0.9, "center_y": 0.175}
        delete_button_5.bind(on_press=self.on_click_delete_button_5)

        new_profile_popup_content = MDFloatLayout()
        new_profile_popup_input = MDTextField(
            hint_text="Enter your name", pos_hint={"center_x": 0.5, "center_y": 0.5}
        )
        new_profile_popup_confirm_button = MDFlatButton(
            text="ADD",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        new_profile_popup_confirm_button.bind(
            on_press=self.on_click_confirm_new_profile_button
        )
        new_profile_popup_cancel_button = MDFlatButton(
            text="CANCEL",
        )
        new_profile_popup_cancel_button.bind(
            on_press=self.on_click_cancel_new_profile_button
        )
        new_profile_popup_content.add_widget(new_profile_popup_input)
        new_profile_popup = MDDialog(
            title="Add new profile:",
            type="custom",
            content_cls=new_profile_popup_content,
            buttons=[
                new_profile_popup_cancel_button,
                new_profile_popup_confirm_button,
            ],
        )

        new_profile_popup.size_hint = (0.6, 0.3)
        new_profile_popup.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        back_button_profile = MDIconButton(icon="reply")
        back_button_profile.bind(on_press=self.on_click_back_button)
        back_button_profile.pos_hint = {"center_x": 0.1, "center_y": 0.96}

        bl_choose_single_profile.add_widget(cie_logo)
        bl_choose_single_profile.add_widget(back_button_profile)
        bl_choose_single_profile.add_widget(profile_button_0)
        bl_choose_single_profile.add_widget(profile_button_1)
        bl_choose_single_profile.add_widget(profile_button_2)
        bl_choose_single_profile.add_widget(profile_button_3)
        bl_choose_single_profile.add_widget(profile_button_4)
        bl_choose_single_profile.add_widget(profile_button_5)
        # bl_choose_single_profile.add_widget(new_profile_popup)
        bl_choose_single_profile.add_widget(delete_button_0)
        bl_choose_single_profile.add_widget(delete_button_1)
        bl_choose_single_profile.add_widget(delete_button_2)
        bl_choose_single_profile.add_widget(delete_button_3)
        bl_choose_single_profile.add_widget(delete_button_4)
        bl_choose_single_profile.add_widget(delete_button_5)
        sm.ids["single_profile_button_0"] = profile_button_0
        sm.ids["single_profile_button_1"] = profile_button_1
        sm.ids["single_profile_button_2"] = profile_button_2
        sm.ids["single_profile_button_3"] = profile_button_3
        sm.ids["single_profile_button_4"] = profile_button_4
        sm.ids["single_profile_button_5"] = profile_button_5
        sm.ids["single_new_profile_popup"] = new_profile_popup
        sm.ids["single_new_profile_input"] = new_profile_popup_input
        sm.ids["single_delete_button_0"] = delete_button_0
        sm.ids["single_delete_button_1"] = delete_button_1
        sm.ids["single_delete_button_2"] = delete_button_2
        sm.ids["single_delete_button_3"] = delete_button_3
        sm.ids["single_delete_button_4"] = delete_button_4
        sm.ids["single_delete_button_5"] = delete_button_5

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_choose_bo3_profiles = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        bo3_switch_0 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.8})
        bo3_switch_1 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.675})
        bo3_switch_2 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.55})
        bo3_switch_3 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.425})
        bo3_switch_4 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.3})
        bo3_switch_5 = MDSwitch(active=True, pos_hint={"x": 0.8, "y": 0.175})
        self.color_prefix = [
            "#007599",
            "#BA2525",
            "#ffddfb",
            "#FFEE00",
            "#82ffff",
            "#fd9b1c",
        ]
        bo3_switch_label_0 = MDLabel(
            text=self.names[0],
            pos_hint={"center_x": 0.3, "center_y": 0.8},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        bo3_switch_label_1 = MDLabel(
            text=self.names[1],
            pos_hint={"x": 0.3, "y": 0.675},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        bo3_switch_label_2 = MDLabel(
            text=self.names[2],
            pos_hint={"x": 0.3, "y": 0.55},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        bo3_switch_label_3 = MDLabel(
            text=self.names[3],
            pos_hint={"x": 0.3, "y": 0.425},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        bo3_switch_label_4 = MDLabel(
            text=self.names[4],
            pos_hint={"x": 0.3, "y": 0.3},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        bo3_switch_label_5 = MDLabel(
            text=self.names[5],
            pos_hint={"center_x": 0.3, "center_y": 0.175},
            size_hint=(0.4, 0.1),
            font_style="H6",
            markup=True,
        )
        start_bo3_session_button = MDRectangleFlatButton(
            text="Start", font_style="H6", line_width=1.2
        )
        start_bo3_session_button.bind(
            on_press=self.on_click_choose_bo3_profile_to_count_button
        )
        start_bo3_session_button.size_hint = (0.3, 0.05)
        start_bo3_session_button.line_color = "#dddddd"
        start_bo3_session_button.pos_hint = {"center_x": 0.5, "center_y": 0.05}
        back_button_b03_profiles = MDIconButton(icon="reply")
        back_button_b03_profiles.bind(on_press=self.on_click_back_button)
        # back_button_b03_profiles.size_hint = (0.05, 0.05)
        back_button_b03_profiles.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button_b03_profiles.background_color = self.theme_cls.primary_color
        # back_button_b03_profiles.color = (1, 1, 1, 1)

        bl_choose_bo3_profiles.add_widget(cie_logo)
        bl_choose_bo3_profiles.add_widget(back_button_b03_profiles)
        bl_choose_bo3_profiles.add_widget(bo3_switch_0)
        bl_choose_bo3_profiles.add_widget(bo3_switch_1)
        bl_choose_bo3_profiles.add_widget(bo3_switch_2)
        bl_choose_bo3_profiles.add_widget(bo3_switch_3)
        bl_choose_bo3_profiles.add_widget(bo3_switch_4)
        bl_choose_bo3_profiles.add_widget(bo3_switch_5)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_0)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_1)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_2)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_3)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_4)
        bl_choose_bo3_profiles.add_widget(bo3_switch_label_5)
        bl_choose_bo3_profiles.add_widget(start_bo3_session_button)
        sm.ids["bo3_switch_0"] = bo3_switch_0
        sm.ids["bo3_switch_1"] = bo3_switch_1
        sm.ids["bo3_switch_2"] = bo3_switch_2
        sm.ids["bo3_switch_3"] = bo3_switch_3
        sm.ids["bo3_switch_4"] = bo3_switch_4
        sm.ids["bo3_switch_5"] = bo3_switch_5
        sm.ids["bo3_switch_label_0"] = bo3_switch_label_0
        sm.ids["bo3_switch_label_1"] = bo3_switch_label_1
        sm.ids["bo3_switch_label_2"] = bo3_switch_label_2
        sm.ids["bo3_switch_label_3"] = bo3_switch_label_3
        sm.ids["bo3_switch_label_4"] = bo3_switch_label_4
        sm.ids["bo3_switch_label_5"] = bo3_switch_label_5

        resume_popup_button = MDFlatButton(
            text="RESUME",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        resume_popup_button.bind(on_press=self.on_click_resume_popup_button)
        restart_popup_button = MDFlatButton(
            text="RESTART",
        )
        restart_popup_button.bind(on_press=self.on_click_restart_popup_button)

        self.resume_or_restart_popup = MDDialog(
            title="Resume or Restart bo3",
            type="custom",
            buttons=[
                restart_popup_button,
                resume_popup_button,
            ],
            size_hint=(0.8, 0.3),
        )

        acknowledge_no_active_profiles_warning_button = MDFlatButton(
            text="OK",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        acknowledge_no_active_profiles_warning_button.bind(
            on_press=self.on_click_acknowledge_no_active_profiles_warning
        )
        self.no_active_profiles_warning_popup = MDDialog(
            title="Please create profiles first",
            type="custom",
            buttons=[
                acknowledge_no_active_profiles_warning_button,
            ],
            size_hint=(0.8, 0.3),
        )

        acknowledge_name_taken_button = MDFlatButton(
            text="OK",
            theme_text_color="Custom",
            text_color=self.theme_cls.primary_color,
        )
        acknowledge_name_taken_button.bind(
            on_press=self.on_click_acknowledge_name_taken_button
        )
        self.name_taken_popup = MDDialog(
            title="Name already taken",
            type="custom",
            buttons=[
                acknowledge_name_taken_button,
            ],
            size_hint=(0.6, 0.2),
        )

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_camera = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        back_button1 = MDIconButton(icon="reply")
        back_button1.bind(on_press=self.on_click_back_button)
        # back_button1.size_hint = (0.05, 0.05)
        back_button1.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button1.background_color = self.theme_cls.primary_color
        # back_button1.color = (1, 1, 1, 1)
        capture_button = MDIconButton(icon="camera")
        capture_button.user_font_size = "56dp"
        # capture_button.size_hint = (0.2, None)
        capture_button.pos_hint = {"center_x": 0.5, "center_y": 0.05}
        self.c = 1
        self.active_profile_index = 0
        capture_button.bind(on_press=self.on_click_capture_button)

        sm.cam = CustomPreview(aspect_ratio="1:1")
        sm.cam.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        sm.cam.size_hint = (1.0, 1.0)
        sm.cam.keep_ratio = True
        sm.cam.allow_stretch = True
        sm.cam.im_shape = [10, 10]

        self.cam_ratio = 1
        bl_camera.add_widget(cie_logo)
        bl_camera.add_widget(sm.cam)
        bl_camera.add_widget(back_button1)
        bl_camera.add_widget(capture_button)

        bl_camera_overlay = MDBoxLayout(orientation="vertical", opacity=0.5)
        overlay = Image(source="mask.jpg")
        overlay.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        overlay.allow_stretch = True
        overlay.keep_ratio = True
        Clock.schedule_interval(self.resize_overlay, 1)

        bl_camera_overlay.add_widget(overlay)
        sm.ids["overlay"] = overlay

        bl_camera_warning = MDBoxLayout(orientation="vertical")
        warning_label = MDLabel(
            text="",
            halign="center",
            valign="center",
            font_style="H5",
            theme_text_color="Error",
        )
        bl_camera_warning.add_widget(warning_label)
        sm.ids["warning"] = warning_label

        bl_counting = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        score_label = MDLabel(
            text="0",
            halign="center",
            valign="center",
            markup=True,
            font_style="H1",
            pos_hint={"center_x": 0.5, "center_y": 0.55},
        )
        highscore_label = MDLabel(
            text="",
            halign="center",
            valign="center",
            markup=True,
            pos_hint={"center_x": 0.5, "center_y": 0.4},
        )
        recount_button = MDRectangleFlatButton(
            text="Recount",
        )
        recount_button.bind(on_press=self.on_click_recount_button)
        recount_button.size_hint = (0.4, 0.1)
        recount_button.line_color = "#ffddfb"
        recount_button.pos_hint = {"center_x": 0.5, "center_y": 0.075}
        back_button2 = MDIconButton(icon="reply")
        back_button2.bind(on_press=self.on_click_back_button)
        # back_button2.size_hint = (0.05, 0.05)
        back_button2.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button2.background_color = self.theme_cls.primary_color
        # back_button2.color = (1, 1, 1, 1)

        bl_counting.add_widget(highscore_label)
        bl_counting.add_widget(recount_button)
        bl_counting.add_widget(cie_logo)
        bl_counting.add_widget(back_button2)
        bl_counting.add_widget(score_label)

        self.points = 0
        self.counting_animation = Clock.schedule_interval(
            self._count_update_label, 1.0 / (100.0)
        )

        sm.ids["highscore_label"] = highscore_label
        sm.ids["point_value"] = score_label

        self.get_active_profiles()
        bl_stats = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        stats_table = MDDataTable(
            use_pagination=False,
            size_hint=(0.9, 0.75),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            column_data=[("", dp(30))]
            + [(name, dp(30)) for name in self.active_profile_names],
            row_data=[
                tuple(
                    ["Max"]
                    + [str(self.stats[i][0]) for i in self.active_profile_indices]
                ),
                tuple(
                    ["Avg"]
                    + [str(self.stats[i][1]) for i in self.active_profile_indices]
                ),
                tuple(
                    ["Last"]
                    + [str(self.stats[i][2]) for i in self.active_profile_indices]
                ),
                tuple(
                    ["#"] + [str(self.stats[i][3]) for i in self.active_profile_indices]
                ),
            ],
            rows_num=10,
        )

        back_button3 = MDIconButton(icon="reply")
        back_button3.bind(on_press=self.on_click_back_button)
        # back_button3.size_hint = (0.05, 0.05)
        back_button3.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button3.background_color = self.theme_cls.primary_color
        # back_button3.color = (1, 1, 1, 1)

        bl_stats.add_widget(back_button3)
        bl_stats.add_widget(cie_logo)
        bl_stats.add_widget(stats_table)
        sm.ids["stats_table"] = stats_table
        sm.ids["bl_stats"] = bl_stats

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_info = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        info_label = MDLabel(
            text='[b][color=#fd9b1c]-  Usage Info  -[/color][/b]\n\n[color=#88B358]Hold the phone parallel to the board.[/color]\n[color=#BA2525]Hold the camera centered over the board.[/color]\n[color=#007599]Match the mask and the board borders.[/color]\n[color=#FFEE00]Counted points seem wrong? Just press the "Recount" button.[/color]\n[color=#ff00e6]Press on the camera image to focus it.[/color]\n[color=#dddddd]Change the camera zoom by moving 2 fingers on the image.[/color]',
            halign="center",
            valign="center",
            markup=True,
            theme_text_color="ContrastParentBackground",
        )
        info_label.size_hint = (0.8, 1)
        info_label.pos_hint = {"x": 0.1, "y": 0}
        info_label.halign = "center"
        info_label.valign = "center"
        info_label.color = (1, 1, 1, 1)

        back_button4 = MDIconButton(icon="reply")
        back_button4.bind(on_press=self.on_click_back_button)
        # back_button4.size_hint = (0.05, 0.05)
        back_button4.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        # back_button4.background_color = self.theme_cls.primary_color
        # back_button4.color = (1, 1, 1, 1)

        bl_info.add_widget(cie_logo)
        bl_info.add_widget(back_button4)
        bl_info.add_widget(info_label)

        # =====================================
        # --- Enter message ---
        # =====================================
        bl_bo3_stats = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        bo3_stats_table = MDDataTable()

        back_button = MDIconButton(icon="reply")
        back_button.bind(on_press=self.on_click_back_button)
        back_button.pos_hint = {"center_x": 0.1, "center_y": 0.96}

        bl_bo3_stats.add_widget(back_button)
        bl_bo3_stats.add_widget(cie_logo)
        bl_bo3_stats.add_widget(bo3_stats_table)
        sm.ids["bo3_stats_table"] = bo3_stats_table
        sm.ids["bl_bo3_stats"] = bl_bo3_stats

        bl_settings = MDFloatLayout()
        cie_logo = MDLabel(
            text="[color=#88B358]C[/color][color=#BA2525]o[/color][color=#FFEE00]u[/color][color=#007599]n[/color][color=#ffddfb]t[/color] [color=#82ffff]i[/color][color=#dddddd]t[/color] [color=#fd9b1c]E[/color][color=#ff00e6]a[/color][color=#88B358]s[/color][color=#007599]y[/color][color=#BA2525]![/color]",
            pos_hint={"center_x": 0.5, "center_y": 0.95},
            halign="center",
            font_style="H5",
            markup=True,
            theme_text_color="Primary",
            bold=True,
            italic=True,
            adaptive_height=True,
        )
        settings_label_save_ims = MDLabel(
            text="[color=#82ffff]Save Images[/color]",
            halign="center",
            valign="center",
            markup=True,
        )
        settings_label_save_ims.size_hint = (0.6, 0.1)
        settings_label_save_ims.pos_hint = {"center_x": 0.4, "center_y": 0.5}
        back_button = MDIconButton(icon="reply")
        back_button.bind(on_press=self.on_click_back_button)
        back_button.pos_hint = {"center_x": 0.1, "center_y": 0.96}
        self.settings_switch_save_ims = MDSwitch(
            active=False, pos_hint={"center_x": 0.8, "center_y": 0.5}
        )

        bl_settings.add_widget(back_button)
        bl_settings.add_widget(cie_logo)
        bl_settings.add_widget(settings_label_save_ims)
        bl_settings.add_widget(self.settings_switch_save_ims)

        Clock.schedule_interval(self.connect_back_click, 1)

        screen_home.add_widget(bl_home)
        screen_choose_single_profile.add_widget(bl_choose_single_profile)
        screen_choose_bo3_profiles.add_widget(bl_choose_bo3_profiles)
        screen_choose_bo3_profile_to_count.add_widget(bl_choose_bo3_profile_to_count)
        screen_profile.add_widget(bl_profile)
        screen_camera.add_widget(bl_camera)
        screen_camera.add_widget(bl_camera_overlay)
        screen_camera.add_widget(bl_camera_warning)
        screen_counting.add_widget(bl_counting)
        screen_stats.add_widget(bl_stats)
        screen_bo3_stats.add_widget(bl_bo3_stats)
        screen_info.add_widget(bl_info)
        screen_settings.add_widget(bl_settings)

        sm.add_widget(screen_home)
        sm.add_widget(screen_choose_single_profile)
        sm.add_widget(screen_choose_bo3_profiles)
        sm.add_widget(screen_choose_bo3_profile_to_count)
        sm.add_widget(screen_profile)
        sm.add_widget(screen_camera)
        sm.add_widget(screen_counting)
        sm.add_widget(screen_stats)
        sm.add_widget(screen_bo3_stats)
        sm.add_widget(screen_info)
        sm.add_widget(screen_settings)

        sm.ids["Home"] = screen_home
        sm.ids["Choose_single_profile"] = screen_choose_single_profile
        sm.ids["Choose_bo3_profiles"] = screen_choose_bo3_profiles
        sm.ids["Choose_bo3_profile_to_count"] = screen_choose_bo3_profile_to_count
        sm.ids["Profiles"] = screen_profile
        sm.ids["Camera"] = screen_camera
        sm.ids["Counting"] = screen_counting
        sm.ids["Stats"] = screen_stats
        sm.ids["Bo3_stats"] = screen_bo3_stats
        sm.ids["Info"] = screen_info
        sm.ids["Settings"] = screen_settings

        self.model159 = CustomModel("model159.tflite")
        self.model267 = CustomModel("model267.tflite")
        self.model348 = CustomModel("model348.tflite")

        self.root_widget = sm
        return self.root_widget

    def on_start(self):
        self.dont_gc = AndroidPermissions()

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_the_camera)

    def connect_back_click(self, *args):
        Window.bind(on_keyboard=self.Android_back_click)
        EventLoop.window.bind(on_keyboard=self.Android_back_click)
        return

    def connect_the_camera(self, dt=None):
        self.root_widget.cam.connect_camera(enable_analyze_pixels=True)

    def on_click_info_button(self, *args):
        self.inside_bo3_screens = False
        self.root.current = "Info"

    def on_click_back_button(self, *args):
        if self.root.current == "Camera":
            self.root_widget.cam.disconnect_camera()
        if self.root.current == "Choose_bo3_profile_to_count":
            self.inside_bo3_screens = False
        if self.go_to_bo3_stats_table:
            self.go_to_bo3_stats_table = False
            self.inside_bo3_screens = False
            self.root.current = "Bo3_stats"
        elif self.inside_bo3_screens:
            self.go_to_bo3_stats_table = False
            self.root.current = "Choose_bo3_profile_to_count"
        else:
            self.go_to_bo3_stats_table = False
            self.root.current = "Home"
        self.go_to_bo3_stats_table = False

    def on_click_back_home_button(self, *args):
        self.root.current = "Home"

    def on_click_recount_button(self, *args):
        if self.active_bo3_session:
            self.root_widget.ids[
                f"bo3_profile_button_{self.active_profile_index}"
            ].opacity = 1
            self.root_widget.ids[
                f"bo3_profile_button_{self.active_profile_index}"
            ].disabled = False
            if (
                self.bo3_session_scores[self.active_profile_index, self.bo3_round]
                != self.points
            ):
                self.bo3_round -= 1
                for idx in self.active_bo3_profile_indices:
                    if idx == self.active_profile_index:
                        continue
                    self.root_widget.ids[f"bo3_profile_button_{idx}"].opacity = 0.5
                    self.root_widget.ids[f"bo3_profile_button_{idx}"].disabled = True
                if self.bo3_round == 2:
                    self.inside_bo3_screens = True
                    self.go_to_bo3_stats_table = False

            self.bo3_session_scores[self.active_profile_index, self.bo3_round] = 0

        self.history[self.active_profile_index].pop()
        self.update_stats(new_values=False)
        self.open_camera()

    def on_click_count_it_button(self, *args):
        self.inside_bo3_screens = False
        self.forward_to_counting = True
        self.prefix = "single_"
        self.get_active_profiles()
        self.update_profile_positions()
        self.root.current = "Choose_single_profile"      

    def on_click_bo3_button(self, *args):
        if self.active_bo3_session:
            self.resume_or_restart_popup.open()
        else:
            self.forward_to_counting = False
            self.get_active_profiles()

            if len(self.active_profile_indices) == 0:
                self.no_active_profiles_warning_popup.open()
            else:
                self.update_switches()
                self.root.current = "Choose_bo3_profiles"

    def on_click_acknowledge_no_active_profiles_warning(self, *args):
        self.no_active_profiles_warning_popup.dismiss()

    def on_click_resume_popup_button(self, *args):
        self.prefix = "bo3_"
        self.forward_to_counting = True
        self.inside_bo3_screens = True
        self.get_active_profiles()
        self.get_active_bo3_profile_indices()
        self.update_profile_positions()
        self.resume_or_restart_popup.dismiss()
        self.root.current = "Choose_bo3_profile_to_count"

    def on_click_restart_popup_button(self, *args):
        self.active_bo3_session = False
        self.forward_to_counting = False
        self.inside_bo3_screens = False
        self.get_active_profiles()
        self.update_switches()
        self.bo3_session_scores = np.zeros((6, 3))
        self.update_stats(new_values=False)
        self.resume_or_restart_popup.dismiss()
        self.root.current = "Choose_bo3_profiles"

    def on_click_choose_bo3_profile_to_count_button(self, *args):
        self.get_active_bo3_profile_indices()
        if len(self.active_bo3_profile_indices) > 0:
            self.prefix = "bo3_"
            self.inside_bo3_screens = True
            self.update_profile_positions()
            self.forward_to_counting = True
            self.active_bo3_session = True
            self.bo3_round = 0
            self.bo3_session_scores = np.zeros((6, 3))
            self.root.current = "Choose_bo3_profile_to_count"

    def on_click_profiles_button(self, *args):
        self.prefix = "profiles_"
        self.inside_bo3_screens = False
        self.forward_to_counting = False
        self.update_profile_positions()
        self.root.current = "Profiles"

    def update_profile_positions(self, *args):
        self.get_active_profiles()
        cy = 0.8
        for index in range(6):
            if self.prefix == "bo3_":
                if index in self.active_bo3_profile_indices:
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].pos_hint = {
                    "center_x": 0.5,
                    "center_y": cy,
                    }

                    if self.bo3_session_scores[index, self.bo3_round] == 0:
                        self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 1
                        self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = False
                    else:
                        self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 0.5
                        self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = True

                    cy -= 0.125
                else:
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 0
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = True

                    
            else:
                if index in self.active_profile_indices:
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].pos_hint = {
                        "center_x": 0.5,
                        "center_y": cy,
                    }
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 1
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = False
                   
                    self.root_widget.ids[f"{self.prefix}delete_button_{index}"].pos_hint = {
                        "center_x": 0.9,
                        "center_y": cy,
                    }
                    self.root_widget.ids[f"{self.prefix}delete_button_{index}"].opacity = 1
                    self.root_widget.ids[f"{self.prefix}delete_button_{index}"].disabled = False

                    cy -= 0.125
            
                else:
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 0
                    self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = True
                    self.root_widget.ids[f"{self.prefix}delete_button_{index}"].opacity = 0
                    self.root_widget.ids[f"{self.prefix}delete_button_{index}"].disabled = True

        if (len(self.active_profile_indices)<6) and (not self.prefix == "bo3_"):
            if not 0 in self.active_profile_indices:
                index = 0
            elif not 1 in self.active_profile_indices:
                index = 1
            elif not 2 in self.active_profile_indices:
                index = 2
            elif not 3 in self.active_profile_indices:
                index = 3
            elif not 4 in self.active_profile_indices:
                index = 4
            elif not 5 in self.active_profile_indices:
                index = 5
            self.root_widget.ids[f"{self.prefix}profile_button_{index}"].pos_hint = {
                    "center_x": 0.5,
                    "center_y": cy,
                }
            self.root_widget.ids[f"{self.prefix}profile_button_{index}"].opacity = 1
            self.root_widget.ids[f"{self.prefix}profile_button_{index}"].disabled = False
            self.root_widget.ids[f"{self.prefix}delete_button_{index}"].opacity = 0
            self.root_widget.ids[f"{self.prefix}delete_button_{index}"].disabled = True

    def open_camera(self, *args):
        self.points = 0
        self.c = 1
        self.root_widget.cam.connect_camera(enable_analyze_pixels=True)
        self.root.current = "Camera"

    def resize_overlay(self, *args):
        self.cam_resolution = self.root_widget.cam.im_shape
        self.cam_ratio = self.cam_resolution[1] / self.cam_resolution[0]
        if self.cam_ratio > 1:
            self.root_widget.ids["overlay"].size_hint = (
                0.9 * 1 / self.cam_ratio,
                0.9 * 1,
            )
        else:
            self.root_widget.ids["overlay"].size_hint = (
                0.9 * 1,
                0.9 * 1 / self.cam_ratio,
            )

    def on_click_settings_button(self, *args):
        self.inside_bo3_screens = False
        self.root.current = "Settings"

    def on_click_capture_button(self, *args):
        self.c = 1
        self.points = 0
        self.root_widget.ids["highscore_label"].text = ""
        im = self.root_widget.cam.current_image
        if self.settings_switch_save_ims.active:
            self.root_widget.cam.capture_photo()

        pred159, pred267, pred348 = self.predict_on_img(im)
        max_occurence = np.max(
            [
                np.unique(p, return_counts=True)[1].max()
                for p in [pred159, pred267, pred348]
            ]
        )
        # pred159, pred267, pred348 = np.random.randint(0, 3, size=(3, 19))
        self.points = self.count_points(pred159, pred267, pred348)
        # max_occurence = 0
        # self.points = 183
        self.point_string = str(self.points)
        if max_occurence > 9 or self.points == 0:
            self.root_widget.ids[
                "warning"
            ].text = "counting did not work\nplease try again"
            self.hide_warning_schedule = Clock.schedule_once(self._hide_warning, 3)
        else:
            if self.active_bo3_session and self.inside_bo3_screens:
                self.update_bo3_session_scores()
            self.update_stats()
            self.root.current = "Counting"

            self.root_widget.cam.disconnect_camera()

    def update_bo3_session_scores(self):
        self.bo3_session_scores[self.active_profile_index][self.bo3_round] = self.points
        if np.all(
            [
                self.bo3_session_scores[idx][self.bo3_round] != 0
                for idx in self.active_bo3_profile_indices
            ]
        ):
            if self.bo3_round == 2:
                self.end_bo3_session()
                self.root_widget.ids[
                    f"bo3_profile_button_{self.active_profile_index}"
                ].opacity = 0.5
                self.root_widget.ids[
                    f"bo3_profile_button_{self.active_profile_index}"
                ].disabled = True
            else:
                self.start_new_bo3_round()
        else:
            self.root_widget.ids[
                f"bo3_profile_button_{self.active_profile_index}"
            ].opacity = 0.5
            self.root_widget.ids[
                f"bo3_profile_button_{self.active_profile_index}"
            ].disabled = True

    def _count_update_label(self, *args):
        if self.c < self.points:
            self.c += np.random.randint(1, 6)
            if self.c > self.points:
                self.c = self.points
            self.rdm_color_idx = np.random.randint(0, 9, size=(3))
            self.point_string = ""
            for i, num in enumerate(str(self.c)):
                self.point_string += f"[b][color={self.counting_colors[self.rdm_color_idx[i]]}]{num}[/color][/b]"

            if self.c == self.highscore_values[self.active_profile_index]:
                self.root_widget.ids[
                    "highscore_label"
                ].text = f"[color={self.counting_colors[np.random.randint(0,9)]}]\n\n!! NEW PERSONAL HIGHSCORE !![/color]"

            self.root_widget.ids["point_value"].text = self.point_string

    def start_new_bo3_round(self):
        self.bo3_round += 1
        self.root_widget.ids[
            "bo3_round_label"
        ].text = (
            f"[color={self.round_label_color_prefix}]Round {self.bo3_round+1}[/color]"
        )
        for index in self.active_bo3_profile_indices:
            self.root_widget.ids[f"bo3_profile_button_{index}"].opacity = 1
            self.root_widget.ids[f"bo3_profile_button_{index}"].disabled = False

    def end_bo3_session(self):
        self.inside_bo3_screens = False
        self.go_to_bo3_stats_table = True
        self.update_bo3_stats_table()

    def on_click_bo3_stats_button(self, *args):
        self.inside_bo3_screens = True
        self.update_bo3_stats_table()
        self.root.current = "Bo3_stats"

    def on_click_stats_button(self, *args):
        self.inside_bo3_screens = False
        self.update_stats_table()
        self.root.current = "Stats"

    def on_click_profile_button_0(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_0"].text == "Add Profile":
            self.new_profile_index = 0
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 0
            self.active_profile_name = self.names[0]
            self.open_camera()

    def on_click_profile_button_1(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_1"].text == "Add Profile":
            self.new_profile_index = 1
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 1
            self.active_profile_name = self.names[1]
            self.open_camera()

    def on_click_profile_button_2(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_2"].text == "Add Profile":
            self.new_profile_index = 2
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 2
            self.active_profile_name = self.names[2]
            self.open_camera()

    def on_click_profile_button_3(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_3"].text == "Add Profile":
            self.new_profile_index = 3
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 3
            self.active_profile_name = self.names[3]
            self.open_camera()

    def on_click_profile_button_4(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_4"].text == "Add Profile":
            self.new_profile_index = 4
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 4
            self.active_profile_name = self.names[4]
            self.open_camera()

    def on_click_profile_button_5(self, *args):
        if self.root_widget.ids[f"{self.prefix}profile_button_5"].text == "Add Profile":
            self.new_profile_index = 5
            self.root_widget.ids[f"{self.prefix}new_profile_popup"].open()
        elif self.forward_to_counting:
            self.active_profile_index = 5
            self.active_profile_name = self.names[5]
            self.open_camera()

    def on_click_delete_button_0(self, *args):
        self.delete_profile_index = 0
        self.confirm_delete_popup.open()

    def on_click_delete_button_1(self, *args):
        self.delete_profile_index = 1
        self.confirm_delete_popup.open()

    def on_click_delete_button_2(self, *args):
        self.delete_profile_index = 2
        self.confirm_delete_popup.open()

    def on_click_delete_button_3(self, *args):
        self.delete_profile_index = 3
        self.confirm_delete_popup.open()

    def on_click_delete_button_4(self, *args):
        self.delete_profile_index = 4
        self.confirm_delete_popup.open()

    def on_click_delete_button_5(self, *args):
        self.delete_profile_index = 5
        self.confirm_delete_popup.open()

    def on_click_confirm_new_profile_button(self, *args):
        self.get_active_profiles()
        new_name = self.root_widget.ids[f"{self.prefix}new_profile_input"].text
        if new_name == "":
            return
        elif new_name in self.active_profile_names:
            self.name_taken_popup.open()
            self.root_widget.ids[f"{self.prefix}new_profile_input"].text = ""
            return
        self.root_widget.ids[
            f"profiles_profile_button_{self.new_profile_index}"
        ].text = new_name
        self.root_widget.ids[
            f"single_profile_button_{self.new_profile_index}"
        ].text = new_name
        self.root_widget.ids[
            f"bo3_profile_button_{self.new_profile_index}"
        ].text = new_name
        self.root_widget.ids[
            f"bo3_switch_label_{self.new_profile_index}"
        ].text = new_name
        self.names[self.new_profile_index] = new_name
        self.highscore_values[self.new_profile_index] = 0
        self.history[self.new_profile_index] = [0]
        self.get_active_profiles()
        self.update_stats()
        self.update_profile_positions()
        self.root_widget.ids[f"{self.prefix}new_profile_input"].text = ""
        self.root_widget.ids[f"{self.prefix}new_profile_popup"].dismiss()
        self.active_profile_index = self.new_profile_index
        self.active_profile_name = self.names[self.new_profile_index]
        if self.forward_to_counting:
            self.open_camera()

    def on_click_acknowledge_name_taken_button(self, *args):
        self.name_taken_popup.dismiss()

    def on_click_cancel_new_profile_button(self, *args):
        self.root_widget.ids[f"{self.prefix}new_profile_input"].text = ""
        self.root_widget.ids[f"{self.prefix}new_profile_popup"].dismiss()

    def on_click_confirm_delete_button(self, *args):
        self.highscore_values[self.delete_profile_index] = 0
        self.names[self.delete_profile_index] = f"New{self.delete_profile_index}"
        self.active_profile_index = self.delete_profile_index
        self.stats[self.delete_profile_index] = np.zeros_like(self.stats[0])
        self.history[self.delete_profile_index] = [0]
        self.root_widget.ids[
            f"profiles_profile_button_{self.delete_profile_index}"
        ].text = "Add Profile"
        self.root_widget.ids[
            f"single_profile_button_{self.delete_profile_index}"
        ].text = "Add Profile"
        self.root_widget.ids[
            f"bo3_profile_button_{self.delete_profile_index}"
        ].text = "Add Profile"
        self.root_widget.ids[
            f"bo3_switch_label_{self.delete_profile_index}"
        ].text = "Add Profile"
        if self.delete_profile_index in self.active_bo3_profile_indices:
            self.active_bo3_session = False
            self.bo3_session_scores = np.zeros((6, 3))
        self.update_stats(new_values=False)
        self.update_profile_positions()
        self.confirm_delete_popup.dismiss()

    def on_click_cancel_delete_button(self, *args):
        self.confirm_delete_popup.dismiss()

    def _hide_warning(self, *args):
        self.root_widget.ids["warning"].text = ""

    def update_stats(self, new_values=True, *args):
        if new_values:
            if self.points > self.highscore_values[self.active_profile_index]:
                self.highscore_values[self.active_profile_index] = self.points
            if self.history[self.active_profile_index][0] == 0:
                self.history[self.active_profile_index][0] = self.points
            else:
                self.history[self.active_profile_index].append(self.points)

        self.get_active_profiles()
        self.stats[self.active_profile_index][0] = np.max(
            self.history[self.active_profile_index]
        )
        self.stats[self.active_profile_index][1] = np.mean(
            self.history[self.active_profile_index]
        )
        self.stats[self.active_profile_index][2] = self.history[
            self.active_profile_index
        ][-1]
        self.stats[self.active_profile_index][3] = len(
            np.nonzero(self.history[self.active_profile_index])[0]
        )
        np.savetxt(
            f"history_{self.active_profile_index}.txt",
            np.array(self.history[self.active_profile_index]),
            delimiter=",",
        )
        np.savetxt("stats.txt", self.stats, delimiter=",")
        np.savetxt(
            "highscores.txt",
            np.array([v for v in self.highscore_values]),
            header=str(self.names).replace("'", "").replace('"', "")[1:-1],
            delimiter=",",
        )
        np.savetxt("bo3_session_scores.txt", self.bo3_session_scores, delimiter=",")
        bo3_indices = np.zeros((6))
        bo3_indices[self.active_bo3_profile_indices] = 1
        np.savetxt("bo3_session_indices.txt", bo3_indices, delimiter=",")

    def get_active_profiles(self, *args):
        self.active_profile_names = []
        self.active_profile_indices = []
        for i, name in enumerate(self.names):
            if not (name == f"New{i}"):
                self.active_profile_names.append(name)
                self.active_profile_indices.append(i)

    def update_switches(self, *args):
        cy = 0.8
        for index in range(6):
            if index in self.active_profile_indices:
                self.root_widget.ids[f"bo3_switch_label_{index}"].pos_hint = {
                    "center_x": 0.4,
                    "center_y": cy,
                }
                self.root_widget.ids[
                    f"bo3_switch_label_{index}"
                ].text = (
                    f"[color={self.color_prefix[index]}]{self.names[index]}[/color]"
                )
                # self.root_widget.ids[f"bo3_switch_label_{index}"].opacity = 1
                self.root_widget.ids[f"bo3_switch_{index}"].pos_hint = {
                    "center_x": 0.8,
                    "center_y": cy,
                }
                # self.root_widget.ids[f"bo3_switch_{index}"].opacity = 1
                # self.root_widget.ids[f"bo3_switch_{index}"].disabled = False
                cy -= 0.125
            else:
                # self.root_widget.ids[f"bo3_switch_label_{index}"].text = ""
                self.root_widget.ids[f"bo3_switch_label_{index}"].pos_hint = {
                    "x": -1,
                    "y": -1,
                }
                # self.root_widget.ids[f"bo3_switch_label_{index}"].opacity = 0
                self.root_widget.ids[f"bo3_switch_{index}"].pos_hint = {
                    "x": -1,
                    "y": -1,
                }

    def apply_overlay_mask(self, im):
        if im.shape[0] > im.shape[1]:
            cropped_im = im[
                int((im.shape[0] - 0.9 * im.shape[1]) / 2) : int(
                    im.shape[0] - (im.shape[0] - 0.9 * im.shape[1]) / 2
                ),
                int(im.shape[1] * 0.05) : int(im.shape[1] * 0.95),
            ]
        else:
            cropped_im = im[
                int(im.shape[0] * 0.05) : int(im.shape[0] * 0.95),
                int((im.shape[1] - 0.9 * im.shape[0]) / 2) : int(
                    im.shape[1] - (im.shape[1] - 0.9 * im.shape[0]) / 2
                ),
            ]
        return cropped_im

    def predict_on_img(self, im):
        im = self.apply_overlay_mask(im)
        ims = self.apply_num_masks(im, self.get_num_masks())
        ims = self._rescale_batch_pure_python(np.array(ims), 128, 128)

        ims159 = ims[:19]
        ims267 = ims[19:38]
        ims348 = ims[38:]

        pred159 = np.argmax(
            np.array(
                [
                    self.model159.predict(x[None, ...])
                    for x in ((np.array(ims159) * (1 / 255)).astype(np.float32))
                ]
            ),
            axis=-1,
        )
        pred267 = np.argmax(
            np.array(
                [
                    self.model267.predict(x[None, ...])
                    for x in ((np.array(ims267) * (1 / 255)).astype(np.float32))
                ]
            ),
            axis=-1,
        )
        pred348 = np.argmax(
            np.array(
                [
                    self.model348.predict(x[None, ...])
                    for x in ((np.array(ims348) * (1 / 255)).astype(np.float32))
                ]
            ),
            axis=-1,
        )

        return pred159, pred267, pred348

    def count_points(self, pred159, pred267, pred348):
        points = 0
        if pred159[0:3].min() == pred159[0:3].max():
            if pred159[0] == 0:
                points += 3
            elif pred159[0] == 1:
                points += 15
            else:
                points += 27

        if pred159[3:7].min() == pred159[3:7].max():
            if pred159[3] == 0:
                points += 4
            elif pred159[3] == 1:
                points += 20
            else:
                points += 36

        if pred159[7:12].min() == pred159[7:12].max():
            if pred159[7] == 0:
                points += 5
            elif pred159[7] == 1:
                points += 25
            else:
                points += 45

        if pred159[12:16].min() == pred159[12:16].max():
            if pred159[12] == 0:
                points += 4
            elif pred159[12] == 1:
                points += 20
            else:
                points += 36

        if pred159[16:19].min() == pred159[16:19].max():
            if pred159[16] == 0:
                points += 3
            elif pred159[16] == 1:
                points += 15
            else:
                points += 27

        ###

        if pred267[0:3].min() == pred267[0:3].max():
            if pred267[0] == 0:
                points += 6
            elif pred267[0] == 1:
                points += 18
            else:
                points += 21

        if pred267[3:7].min() == pred267[3:7].max():
            if pred267[3] == 0:
                points += 8
            elif pred267[3] == 1:
                points += 24
            else:
                points += 28

        if pred267[7:12].min() == pred267[7:12].max():
            if pred267[7] == 0:
                points += 10
            elif pred267[7] == 1:
                points += 30
            else:
                points += 35

        if pred267[12:16].min() == pred267[12:16].max():
            if pred267[12] == 0:
                points += 8
            elif pred267[12] == 1:
                points += 24
            else:
                points += 28

        if pred267[16:19].min() == pred267[16:19].max():
            if pred267[16] == 0:
                points += 6
            elif pred267[16] == 1:
                points += 18
            else:
                points += 21

        ###

        if pred348[0:3].min() == pred348[0:3].max():
            if pred348[0] == 0:
                points += 9
            elif pred348[0] == 1:
                points += 12
            else:
                points += 24
        if pred348[3:7].min() == pred348[3:7].max():
            if pred348[3] == 0:
                points += 12
            elif pred348[3] == 1:
                points += 16
            else:
                points += 32
        if pred348[7:12].max() == pred348[7:12].min():
            if pred348[7] == 0:
                points += 15
            elif pred348[7] == 1:
                points += 20
            else:
                points += 40
        if pred348[12:16].min() == pred348[12:16].max():
            if pred348[12] == 0:
                points += 12
            elif pred348[12] == 1:
                points += 16
            else:
                points += 32
        if pred348[16:19].min() == pred348[16:19].max():
            if pred348[16] == 0:
                points += 9
            elif pred348[16] == 1:
                points += 12
            else:
                points += 24

        return points

    def get_num_masks(self):
        num_masks = np.array(
            [
                [9.03373356e-02, 1.69811321e-01, 2.56718125e-01, 2.99599771e-01],
                [8.74785592e-02, 3.68782161e-01, 2.53859348e-01, 4.99142367e-01],
                [1.01200686e-01, 5.69468268e-01, 2.67581475e-01, 6.99828473e-01],
                [2.38993711e-01, 8.74785592e-02, 4.05374500e-01, 2.17267010e-01],
                [2.49857061e-01, 2.69868496e-01, 4.16237850e-01, 4.00228702e-01],
                [2.55574614e-01, 4.65980560e-01, 4.21955403e-01, 5.95769011e-01],
                [2.67009720e-01, 6.66094911e-01, 4.33390509e-01, 7.96455117e-01],
                [4.20240137e-01, 5.71755289e-04, 5.86620926e-01, 1.30360206e-01],
                [4.19668382e-01, 1.82389937e-01, 5.85477416e-01, 3.12178388e-01],
                [4.15666095e-01, 3.76214980e-01, 5.82046884e-01, 5.06575186e-01],
                [4.25385935e-01, 5.59176672e-01, 5.91766724e-01, 6.88965123e-01],
                [4.28244711e-01, 7.52429960e-01, 5.94053745e-01, 8.82790166e-01],
                [5.81475129e-01, 8.40480274e-02, 7.47855918e-01, 2.13836478e-01],
                [5.82046884e-01, 2.74442539e-01, 7.48427673e-01, 4.04230989e-01],
                [5.83762150e-01, 4.69982847e-01, 7.50142939e-01, 5.99771298e-01],
                [5.90623213e-01, 6.52944540e-01, 7.57004002e-01, 7.83304746e-01],
                [7.56432247e-01, 1.74957118e-01, 9.22813036e-01, 3.04745569e-01],
                [7.51286449e-01, 3.68210406e-01, 9.17667238e-01, 4.98570612e-01],
                [7.50142939e-01, 5.47169811e-01, 9.16523728e-01, 6.76958262e-01],
                [3.76214980e-01, 6.68953688e-02, 5.42595769e-01, 1.96683819e-01],
                [2.07547170e-01, 1.69811321e-01, 3.73927959e-01, 2.99599771e-01],
                [4.34534019e-02, 2.55574614e-01, 2.09834191e-01, 3.85363065e-01],
                [5.31732419e-01, 1.76100629e-01, 6.98113208e-01, 3.05889079e-01],
                [3.70497427e-01, 2.63007433e-01, 5.36878216e-01, 3.92795883e-01],
                [2.12121212e-01, 3.68210406e-01, 3.78502001e-01, 4.98570612e-01],
                [4.23098914e-02, 4.49399657e-01, 2.08690680e-01, 5.79188107e-01],
                [7.02115495e-01, 2.57289880e-01, 8.68496284e-01, 3.87078330e-01],
                [5.36306461e-01, 3.64208119e-01, 7.02115495e-01, 4.93996569e-01],
                [3.77358491e-01, 4.60834763e-01, 5.43739280e-01, 5.90623213e-01],
                [2.03544883e-01, 5.62035449e-01, 3.69925672e-01, 6.92395655e-01],
                [4.80274443e-02, 6.64379646e-01, 2.14408233e-01, 7.94739851e-01],
                [6.96397942e-01, 4.56832476e-01, 8.62778731e-01, 5.86620926e-01],
                [5.35162950e-01, 5.56317896e-01, 7.01543739e-01, 6.86106346e-01],
                [3.75643225e-01, 6.50085763e-01, 5.42024014e-01, 7.80445969e-01],
                [2.12121212e-01, 7.51858205e-01, 3.78502001e-01, 8.82218411e-01],
                [7.05546026e-01, 6.46083476e-01, 8.71926815e-01, 7.76443682e-01],
                [5.35162950e-01, 7.47284162e-01, 7.01543739e-01, 8.77644368e-01],
                [3.69925672e-01, 8.42767296e-01, 5.36306461e-01, 9.73127501e-01],
                [4.65408805e-01, 7.31846770e-02, 6.31789594e-01, 2.02973128e-01],
                [6.24928531e-01, 1.77815895e-01, 7.91309320e-01, 3.07604345e-01],
                [7.95883362e-01, 2.69296741e-01, 9.62264151e-01, 3.99085192e-01],
                [3.10463122e-01, 1.74385363e-01, 4.76843911e-01, 3.04173814e-01],
                [4.65408805e-01, 2.72727273e-01, 6.31789594e-01, 4.02515723e-01],
                [6.30074328e-01, 3.65351630e-01, 7.96455117e-01, 4.95140080e-01],
                [7.99313894e-01, 4.63693539e-01, 9.65694683e-01, 5.93481990e-01],
                [1.45797599e-01, 2.70440252e-01, 3.12178388e-01, 4.00800457e-01],
                [3.11606632e-01, 3.69925672e-01, 4.77987421e-01, 5.00285878e-01],
                [4.77415666e-01, 4.60834763e-01, 6.43796455e-01, 5.90623213e-01],
                [6.33504860e-01, 5.54030875e-01, 7.99885649e-01, 6.83819325e-01],
                [7.97026872e-01, 6.45511721e-01, 9.63407662e-01, 7.75871927e-01],
                [1.37793025e-01, 4.59691252e-01, 3.04173814e-01, 5.89479703e-01],
                [3.04173814e-01, 5.63750715e-01, 4.70554603e-01, 6.94110921e-01],
                [4.71698113e-01, 6.59805603e-01, 6.38078902e-01, 7.90165809e-01],
                [6.40937679e-01, 7.45568897e-01, 8.07318468e-01, 8.75929102e-01],
                [1.49799886e-01, 6.63236135e-01, 3.16180675e-01, 7.93596341e-01],
                [3.17895941e-01, 7.50714694e-01, 4.84276730e-01, 8.81074900e-01],
                [4.82561464e-01, 8.41623785e-01, 6.48942253e-01, 9.71983991e-01],
            ]
        )

        return num_masks

    def apply_num_masks(self, image, num_masks):
        num_masks = np.round(
            (
                num_masks
                * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
            )
        ).astype(int)
        num_masks[1:, 2] = num_masks[1:, 0] + (num_masks[0, 2] - num_masks[0, 0])
        num_masks[1:, 3] = num_masks[1:, 1] + (num_masks[0, 3] - num_masks[0, 1])
        masked_images = np.zeros(
            (
                57,
                num_masks[0, 3] - num_masks[0, 1],
                num_masks[0, 2] - num_masks[0, 0],
                3,
            )
        )
        for i in range(len(num_masks)):
            masked_images[i, :, :, :] = image[
                num_masks[i, 1] : num_masks[i, 3],
                num_masks[i, 0] : num_masks[i, 2],
                :,
            ]
        return masked_images

    def _rescale_pure_python(self, im, h, w):
        return np.array(
            [
                [
                    im[int(im.shape[0] * r / h)][int(im.shape[1] * c / w)]
                    for c in range(w)
                ]
                for r in range(h)
            ]
        )

    def _rescale_batch_pure_python(self, ims, h, w):
        new_ims = np.zeros((57, w, h, 3))
        for r in range(h):
            for c in range(w):
                new_ims[:, r, c, :] = ims[
                    :,
                    int(ims.shape[1] * r / h),
                    int(ims.shape[2] * c / w),
                    :,
                ]
        return new_ims

    def get_active_bo3_profile_indices(self):
        self.active_bo3_profile_indices = []
        for index in range(6):
            if (
                self.root_widget.ids[f"bo3_switch_{index}"].active
                and index in self.active_profile_indices
            ):
                self.active_bo3_profile_indices.append(index)

    def update_stats_table(self):
        stats_table = MDDataTable(
            use_pagination=False,
            size_hint=(0.9, 0.75),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            column_data=[("", dp(8))]
            + [
                (name, dp(max(8, int(2.5 * len(name)))))
                for name in self.active_profile_names
            ],
            row_data=[
                tuple(
                    ["Max"]
                    + [str(int(self.stats[i][0])) for i in self.active_profile_indices]
                ),
                tuple(
                    ["Avg"]
                    + [str(int(self.stats[i][1])) for i in self.active_profile_indices]
                ),
                tuple(
                    ["Last"]
                    + [str(int(self.stats[i][2])) for i in self.active_profile_indices]
                ),
                tuple(
                    ["#"]
                    + [str(int(self.stats[i][3])) for i in self.active_profile_indices]
                ),
            ],
            rows_num=10,
        )
        self.root_widget.ids["bl_stats"].remove_widget(
            self.root_widget.ids["stats_table"]
        )
        self.root_widget.ids["bl_stats"].add_widget(stats_table)
        self.root_widget.ids["stats_table"] = stats_table

    def update_bo3_stats_table(self):
        highlighting = []
        non_zero_rounds = []
        for i in range(6):
            highlighting_iter = []
            non_zero_rounds_iter = []
            for r in range(3):
                if (
                    self.bo3_session_scores[i, r] == self.bo3_session_scores[:, r].max()
                    and self.bo3_session_scores[i, r] != 0
                ):
                    highlighting_iter.append(["[color=#297B50]", "[/color]"])
                else:
                    highlighting_iter.append(["", ""])
                if self.bo3_session_scores[i, r] != 0:
                    non_zero_rounds_iter.append(self.bo3_session_scores[i, r])
            highlighting_iter.append(["", ""])
            highlighting.append(highlighting_iter)
            non_zero_rounds.append(non_zero_rounds_iter)
        avgs = np.array(
            [
                np.mean(non_zero_rounds[i]) if not len(non_zero_rounds[i]) == 0 else 0
                for i in range(6)
            ]
        )
        if not avgs.max() == 0:
            max_avg = np.argmax(avgs)
            highlighting[max_avg][3] = ["[color=#297B50]", "[/color]"]

        stats_table = MDDataTable(
            use_pagination=False,
            size_hint=(0.9, 0.75),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            column_data=[("", dp(8))]
            + [
                (
                    self.names[idx],
                    dp(max(8, int(2.5 * len((self.names[idx]))))),
                )
                for idx in self.active_bo3_profile_indices
            ],
            row_data=[
                tuple(
                    ["R1"]
                    + [
                        f"{highlighting[idx][0][0]}{str(int(self.bo3_session_scores[idx][0]))}{highlighting[idx][0][1]}"
                        for idx in self.active_bo3_profile_indices
                    ]
                ),
                tuple(
                    ["R2"]
                    + [
                        f"{highlighting[idx][1][0]}{str(int(self.bo3_session_scores[idx][1]))}{highlighting[idx][1][1]}"
                        for idx in self.active_bo3_profile_indices
                    ]
                ),
                tuple(
                    ["R3"]
                    + [
                        f"{highlighting[idx][2][0]}{str(int(self.bo3_session_scores[idx][2]))}{highlighting[idx][2][1]}"
                        for idx in self.active_bo3_profile_indices
                    ]
                ),
                tuple(
                    ["Avg"]
                    + [
                        f"{highlighting[idx][3][0]}{str(int(np.round(avgs[idx])))}{highlighting[idx][3][1]}"
                        for idx in self.active_bo3_profile_indices
                    ]
                ),
            ],
            rows_num=10,
        )
        self.root_widget.ids["bl_bo3_stats"].remove_widget(
            self.root_widget.ids["bo3_stats_table"]
        )
        self.root_widget.ids["bl_bo3_stats"].add_widget(stats_table)
        self.root_widget.ids["bo3_stats_table"] = stats_table


if __name__ == "__main__":
    CountItEasy().run()
