from tkinter import Tk, Button, Canvas, Menu, CENTER, filedialog, Label, messagebox
from FrameNetwork import FrameNetwork
from DataLoader import DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageTk


def convert_cv2_image_to_imagepk(img, flatten=False):
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    return imgtk


class LoadedFrame:
    def __init__(self, frame, original_frame, interpolated):
        self.frame = frame
        self.original_frame = original_frame
        self.interpolated = interpolated


class MainView:
    def __init__(self, comparison_mode=None, original_window=None):
        self.frame_network = None
        self.data_loader = None
        self.loaded_frames = None
        self.interpolated_frame = None
        self.left_frame_index = 0
        self.interpolated_label = None

        self.comparison_mode = comparison_mode

        if original_window:
            for widget in original_window.winfo_children():
                widget.destroy()

        self.width = 1280
        self.height = 720
        self.window = original_window or Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Gerador de quadros intermediários")

        self.paint_canvas()

        self.add_screen_elements()
        self.add_toolbar()

        self.build_network()

    def paint_canvas(self):
        if self.loaded_frames is not None:
            if self.comparison_mode == "toggle":
                self.create_frame_label(self.loaded_frames[self.left_frame_index]).place(relx=0.25, rely=0.5,
                                                                                         anchor=CENTER)
                self.create_frame_label(self.loaded_frames[self.left_frame_index + 1]).place(relx=0.5, rely=0.5,
                                                                                             anchor=CENTER)
                self.create_frame_label(self.loaded_frames[self.left_frame_index + 2]).place(relx=0.75, rely=0.5,
                                                                                             anchor=CENTER)
            elif self.comparison_mode == "side":
                self.create_frame_label(self.loaded_frames[self.left_frame_index]).place(relx=0.25, rely=0.4,
                                                                                         anchor=CENTER)
                self.create_frame_label(self.loaded_frames[self.left_frame_index + 1]).place(relx=0.5, rely=0.2,
                                                                                             anchor=CENTER)
                self.create_frame_label(self.loaded_frames[self.left_frame_index + 2]).place(relx=0.75, rely=0.4,
                                                                                             anchor=CENTER)
            else:
                self.create_frame_label(self.loaded_frames[self.left_frame_index]).place(relx=0.25, rely=0.2,
                                                                                         anchor=CENTER)
                self.create_frame_label(self.loaded_frames[self.left_frame_index + 1]).place(relx=0.75, rely=0.2,
                                                                                             anchor=CENTER)

        if self.interpolated_frame is not None:
            self.interpolated_label = self.create_frame_label(self.interpolated_frame)
            if self.comparison_mode == "toggle":
                if self.compare_toggle_button.config("relief")[-1] == "sunken":
                    self.interpolated_label.place(relx=0.5, rely=0.5, anchor=CENTER)
            elif self.comparison_mode == "side":
                self.interpolated_label.place(relx=0.5, rely=0.6, anchor=CENTER)
            else:
                self.interpolated_label.place(relx=0.5, rely=0.2, anchor=CENTER)
        else:
            if self.interpolated_label:
                self.interpolated_label.destroy()

    def create_frame_label(self, loaded_frame):
        color = "#37d3ff" if loaded_frame.interpolated else "#000000"
        return Label(self.window, image=loaded_frame.frame, highlightthickness=4, highlightbackground=color)

    def add_screen_elements(self):
        move_left_button = Button(self.window, text="<", height=5, width=5,
                                  command=self.move_left_button_on_click)
        move_left_button.place(relx=0.3, rely=0.80, anchor=CENTER)

        move_right_button = Button(self.window, text=">", height=5, width=5,
                                   command=self.move_right_button_on_click)
        move_right_button.place(relx=0.7, rely=0.80, anchor=CENTER)

        if self.comparison_mode == "toggle":
            self.compare_toggle_button = Button(self.window, text="Mostrar gerado", height=5, width=40,
                                                command=self.compare_toggle_button_on_click,
                                                relief="raised")
            self.compare_toggle_button.place(relx=0.5, rely=0.8, anchor=CENTER)
        elif not self.comparison_mode:
            add_frame_button = Button(self.window, text="Gerar quadro", height=5, width=40,
                                      command=self.add_frame_button_on_click)
            add_frame_button.place(relx=0.5, rely=0.73, anchor=CENTER)

            add_all_frames_button = Button(self.window, text="Gerar quadros para todos os pares", height=5, width=40,
                                           command=self.add_all_frames_button_on_click)
            add_all_frames_button.place(relx=0.5, rely=0.85, anchor=CENTER)

    def add_toolbar(self):
        main_toolbar = Menu(self.window)
        self.window.config(menu=main_toolbar)

        importar_submenu = Menu(main_toolbar)
        main_toolbar.add_cascade(label="Importar", menu=importar_submenu)
        importar_submenu.add_command(label="Pasta", command=self.importar_submenu_on_click)

        if not self.comparison_mode:
            main_toolbar.add_cascade(label="Exportar", command=self.exportar_submenu_on_click)

    def build_network(self):
        self.frame_network = FrameNetwork()

        self.frame_network.load_weights(
            "E:\\GianAwesome\\Facultade\\outside_drive\\douga-keras-pix2pix-anime-optimizes-testbed-2400-atd12k-3.h5")

    def add_frame_button_on_click(self):
        self.generate_frame()

    def generate_frame(self):
        if self.loaded_frames is not None:
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

            self.interpolated_frame = LoadedFrame(convert_cv2_image_to_imagepk(256 * gen, flatten=True), gen, True)

            self.paint_canvas()

            if not self.comparison_mode:
                self.loaded_frames = self.loaded_frames[:self.left_frame_index + 1] + [
                    self.interpolated_frame] + self.loaded_frames[self.left_frame_index + 1:]

    def add_all_frames_button_on_click(self):
        pass

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
            title="Escolha a pasta com os quadros")

        self.data_loader = DataLoader(dataset_name=os.path.dirname(dir_name),
                                      img_res=(self.frame_network.img_rows, self.frame_network.img_cols),
                                      root_dir=dir_name)

        for scene in sorted(os.listdir(dir_name)):
            first_scene = scene
            break

        _, imgs_B, _ = self.data_loader.load_all_image_triplet(first_scene)

        self.loaded_frames = [LoadedFrame(convert_cv2_image_to_imagepk(256 * img), img, False) for img in imgs_B]

        if self.comparison_mode:
            self.generate_frame()

        self.paint_canvas()

    def exportar_submenu_on_click(self):
        dir_name = filedialog.askdirectory(initialdir="/",
                                           title="Escolha a pasta para salvar os quadros")

        if dir_name:
            for i, loaded_frame in enumerate(self.loaded_frames):
                cv2.imwrite(os.path.join(dir_name, str(i).zfill(3) + ".png"), loaded_frame.original_frame * 256)

    def open(self):
        self.window.mainloop()
