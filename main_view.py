from tkinter import Tk, Button, Canvas, Menu, CENTER, filedialog, Label, messagebox, Frame
from FrameNetwork import FrameNetwork
from DataLoader import DataLoader
import os
import time
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
from preview_view import PreviewView


def convert_cv2_image_to_imagepk(img, flatten=False):
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    return imgtk


class LoadedFrame:
    def __init__(self, frame, original_frame, name, interpolated):
        self.frame = frame
        self.original_frame = original_frame
        self.interpolated = interpolated
        self.name = name


class MainView:
    def __init__(self, comparison_mode=None, original_window=None):
        self.frame_network = None
        self.data_loader = None
        self.loaded_frames = None
        self.interpolated_frame = None
        self.left_frame_index = 0
        self.interpolated_label = None
        self.generated_frames = {}

        self.comparison_mode = comparison_mode

        if original_window:
            for widget in original_window.winfo_children():
                widget.destroy()

        self.width = 1280
        self.height = 720
        self.window = original_window or Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Inbetween frame generator")

        self.paint_canvas()

        self.add_screen_elements()
        self.add_toolbar()

        self.build_network()

    def paint_canvas(self):
        if self.loaded_frames is not None:
            if self.comparison_mode == "toggle":
                self.place_frame_label(self.loaded_frames[self.left_frame_index], relx=0.25, rely=0.5)
                self.place_frame_label(self.loaded_frames[self.left_frame_index + 1], relx=0.5, rely=0.5)
                self.place_frame_label(self.loaded_frames[self.left_frame_index + 2], relx=0.75, rely=0.5)
            elif self.comparison_mode == "side":
                self.place_frame_label(self.loaded_frames[self.left_frame_index], relx=0.25, rely=0.4)
                self.place_frame_label(self.loaded_frames[self.left_frame_index + 1], relx=0.5, rely=0.6)
                self.place_frame_label(self.loaded_frames[self.left_frame_index + 2], relx=0.75, rely=0.4)
            else:
                self.place_frame_label(self.loaded_frames[self.left_frame_index], relx=0.25, rely=0.2)
                self.place_frame_label(self.loaded_frames[self.left_frame_index + 1], relx=0.75, rely=0.2)

        if self.interpolated_frame is not None:
            self.interpolated_label = self.create_frame_label(self.interpolated_frame)
            if self.comparison_mode == "toggle":
                if self.compare_toggle_button.config("relief")[-1] == "sunken":
                    self.interpolated_label.place(relx=0.5, rely=0.5, anchor=CENTER)
            elif self.comparison_mode == "side":
                self.interpolated_label.place(relx=0.5, rely=0.2, anchor=CENTER)
            else:
                self.interpolated_label.place(relx=0.5, rely=0.2, anchor=CENTER)
        else:
            if self.interpolated_label:
                self.interpolated_label.destroy()

    def place_frame_label(self, loaded_frame, relx=0, rely=0, anchor=CENTER):
        label = self.create_frame_label(loaded_frame)
        label.place(relx=relx, rely=rely, anchor=anchor)

        Label(self.window, text=loaded_frame.name).place(relx=relx, rely=rely+0.21, anchor=anchor)

    def create_frame_label(self, loaded_frame):
        color = "#37d3ff" if loaded_frame.interpolated else "#000000"
        border_frame = Frame(self.window, background=color)

        label = Label(border_frame, image=loaded_frame.frame)
        label.pack(padx=3, pady=3)

        return border_frame

    def add_screen_elements(self):
        move_left_button = Button(self.window, text="<", height=5, width=5,
                                  command=self.move_left_button_on_click)
        move_left_button.place(relx=0.3, rely=0.80, anchor=CENTER)

        move_right_button = Button(self.window, text=">", height=5, width=5,
                                   command=self.move_right_button_on_click)
        move_right_button.place(relx=0.7, rely=0.80, anchor=CENTER)

        if self.comparison_mode == "toggle":
            self.compare_toggle_button = Button(self.window, text="Display generated", height=5, width=40,
                                                command=self.compare_toggle_button_on_click,
                                                relief="raised")
            self.compare_toggle_button.place(relx=0.5, rely=0.8, anchor=CENTER)
        elif not self.comparison_mode:
            add_frame_button = Button(self.window, text="Generate frame", height=5, width=40,
                                      command=self.add_frame_button_on_click)
            add_frame_button.place(relx=0.5, rely=0.73, anchor=CENTER)

            #add_all_frames_button = Button(self.window, text="Gerar quadros para todos os pares", height=5, width=40,
            #                               command=self.add_all_frames_button_on_click)
            #add_all_frames_button.place(relx=0.5, rely=0.85, anchor=CENTER)

            preview_button = Button(self.window, text="Preview animation", height=5, width=40,
                                           command=self.preview_button_on_click)
            preview_button.place(relx=0.5, rely=0.85, anchor=CENTER)

    def add_toolbar(self):
        main_toolbar = Menu(self.window)
        self.window.config(menu=main_toolbar)

        importar_submenu = Menu(main_toolbar)
        main_toolbar.add_cascade(label="Import", menu=importar_submenu)
        importar_submenu.add_command(label="Folder", command=self.importar_submenu_on_click)

        if not self.comparison_mode:
            main_toolbar.add_cascade(label="Export", command=self.exportar_submenu_on_click)

    def build_network(self):
        self.frame_network = FrameNetwork()

        self.frame_network.load_weights(
            "E:\\GianAwesome\\Facultade\\outside_drive\\douga-keras-pix2pix-anime-optimizes-testbed-2400-atd12k-3.h5")

    def add_frame_button_on_click(self):
        self.generate_frame()

    def generate_frame(self):
        if self.loaded_frames is not None:
            if self.comparison_mode and self.left_frame_index in self.generated_frames:
                self.interpolated_frame = self.generated_frames[self.left_frame_index]
            else:
                if self.comparison_mode:
                    gen = self.frame_network.generator.predict(
                        [np.expand_dims(self.loaded_frames[self.left_frame_index].original_frame, axis=0),
                         np.expand_dims(self.loaded_frames[self.left_frame_index + 2].original_frame, axis=0)])
                else:
                    gen = self.frame_network.generator.predict(
                        [np.expand_dims(self.loaded_frames[self.left_frame_index].original_frame, axis=0),
                         np.expand_dims(self.loaded_frames[self.left_frame_index + 1].original_frame, axis=0)])

                gen = gen[0]
                gen = np.squeeze(gen, axis=-1)

                prev_frame = self.loaded_frames[self.left_frame_index]
                name = prev_frame.name + 1 if prev_frame.interpolated else 0
                gen = LoadedFrame(convert_cv2_image_to_imagepk(256 * gen, flatten=True), gen, name, True)
                self.interpolated_frame = gen

                if self.comparison_mode:
                    self.generated_frames[self.left_frame_index] = gen

            self.paint_canvas()

            if not self.comparison_mode:
                self.loaded_frames = self.loaded_frames[:self.left_frame_index + 1] + [
                    self.interpolated_frame] + self.loaded_frames[self.left_frame_index + 1:]

                i = self.left_frame_index + 2
                while self.loaded_frames[i].interpolated:
                    self.loaded_frames[i].name += 1
                    i += 1

    def add_all_frames_button_on_click(self):
        pass

    def preview_button_on_click(self):
        preview_view = PreviewView(self.loaded_frames, master=self.window)

    def move_left_button_on_click(self):
        if self.loaded_frames is not None and self.left_frame_index > 0:
            self.left_frame_index -= 1
            self.interpolated_frame = None

            if self.comparison_mode:
                self.generate_frame()

            self.paint_canvas()

    def move_right_button_on_click(self):
        if self.loaded_frames is not None and self.left_frame_index < len(self.loaded_frames) - 2:
            self.left_frame_index += 1
            self.interpolated_frame = None

            if self.comparison_mode:
                self.generate_frame()

            self.paint_canvas()

    def compare_toggle_button_on_click(self):
        if self.compare_toggle_button.config("relief")[-1] == "sunken":
            self.compare_toggle_button.config(relief="raised")
        else:
            self.compare_toggle_button.config(relief="sunken")

        self.paint_canvas()

    def importar_submenu_on_click(self):
        dir_name = filedialog.askdirectory(
            initialdir="E:\\GianAwesome\\Facultade\\Pesquisa\\Final Phase\\sketch_animated_by_scene",
            title="Choose the frame folder")

        if dir_name:
            self.data_loader = DataLoader(dataset_name=os.path.dirname(dir_name),
                                          img_res=(self.frame_network.img_rows, self.frame_network.img_cols),
                                          root_dir=dir_name)

            _, imgs_B, _ = self.data_loader.load_all_image_triplet()

            self.loaded_frames = [LoadedFrame(convert_cv2_image_to_imagepk(256 * img), img, i, False) for
                                  i, img in enumerate(imgs_B)]

            if self.comparison_mode:
                self.generate_frame()

            self.paint_canvas()

    def exportar_submenu_on_click(self):
        dir_name = filedialog.askdirectory(initialdir="/",
                                           title="Choose the folder to save the frames")

        if dir_name:
            prev_original_index = 0
            number_of_digits = int(math.log10(len(self.loaded_frames))) + 1
            for i, loaded_frame in enumerate(self.loaded_frames):
                if not loaded_frame.interpolated:
                    prev_original_index = loaded_frame.name
                    name = str(prev_original_index).zfill(number_of_digits) + ".png"
                else:
                    name = str(prev_original_index).zfill(number_of_digits) + "-" + str(loaded_frame.name).zfill(
                        number_of_digits) + ".png"

                cv2.imwrite(os.path.join(dir_name, name), loaded_frame.original_frame * 256)

            os.startfile(dir_name)

    def open(self):
        self.window.mainloop()
