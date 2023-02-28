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


class MainView:
    def __init__(self, comparison_mode=None):
        self.frame_network = None
        self.data_loader = None
        self.loaded_frames = None
        self.loaded_original_frames = None
        self.interpolated_frame = None
        self.left_frame_index = 0

        self.comparison_mode = comparison_mode

        self.width = 1280
        self.height = 900
        self.window = Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Gerador de quadros intermediários")

        self.paint_canvas()

        self.add_screen_elements()
        self.add_toolbar()

        self.build_network()

    def paint_canvas(self):
        #canvas = Canvas(self.window, bg='lightgray', height=self.height, width=self.width)
        #canvas.place(relx=0, rely=0)

        #canvas.create_rectangle(100, 150, 405, 300)
        #canvas.create_rectangle(482, 300, 797, 450)
        #canvas.create_rectangle(865, 150, 1180, 300)

        if self.loaded_frames is not None:
            if self.comparison_mode == "toggle":
                Label(self.window, image=self.loaded_frames[self.left_frame_index]).place(relx=0.25, rely=0.2,
                                                                                          anchor=CENTER)
                Label(self.window, image=self.loaded_frames[self.left_frame_index + 1]).place(relx=0.5, rely=0.2,
                                                                                              anchor=CENTER)
                Label(self.window, image=self.loaded_frames[self.left_frame_index + 2]).place(relx=0.75, rely=0.2,
                                                                                              anchor=CENTER)
            elif self.comparison_mode == "side":
                Label(self.window, image=self.loaded_frames[self.left_frame_index]).place(relx=0.17, rely=0.2,
                                                                                          anchor=CENTER)
                Label(self.window, image=self.loaded_frames[self.left_frame_index + 1]).place(relx=0.39, rely=0.2,
                                                                                          anchor=CENTER)
                Label(self.window, image=self.loaded_frames[self.left_frame_index + 2]).place(relx=0.83, rely=0.2,
                                                                                              anchor=CENTER)
            else:
                Label(self.window, image=self.loaded_frames[self.left_frame_index]).place(relx=0.25, rely=0.2,
                                                                                          anchor=CENTER)
                Label(self.window, image=self.loaded_frames[self.left_frame_index+1]).place(relx=0.75, rely=0.2,
                                                                                            anchor=CENTER)

        if self.interpolated_frame is not None:
            if self.comparison_mode == "toggle":
                if self.compare_toggle_button.config("relief")[-1] == "sunken":
                    Label(self.window, image=self.interpolated_frame).place(relx=0.5, rely=0.2, anchor=CENTER)
                else:
                    Label(self.window, image=self.loaded_frames[self.left_frame_index+1])\
                        .place(relx=0.5, rely=0.2, anchor=CENTER)
            elif self.comparison_mode == "side":
                Label(self.window, image=self.interpolated_frame).place(relx=0.61, rely=0.2, anchor=CENTER)
            else:
                Label(self.window, image=self.interpolated_frame).place(relx=0.5, rely=0.2, anchor=CENTER)

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
            self.compare_toggle_button.place(relx=0.5, rely=0.4, anchor=CENTER)
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

        self.frame_network.load_weights("E:\\GianAwesome\\Facultade\\outside_drive\\douga-keras-pix2pix-anime-optimizes-testbed-2400-atd12k-3.h5")

    def add_frame_button_on_click(self):
        self.generate_frame()

    def generate_frame(self):
        if self.loaded_original_frames is not None:
#            gen = self.frame_network.generator.predict([[self.loaded_original_frames[self.left_frame_index]],
#                                                       [self.loaded_original_frames[self.left_frame_index+1]]])

            if self.comparison_mode:
                gen = self.frame_network.generator.predict(
                    [np.expand_dims(self.loaded_original_frames[self.left_frame_index], axis=0),
                     np.expand_dims(self.loaded_original_frames[self.left_frame_index + 2], axis=0)])
            else:
                gen = self.frame_network.generator.predict(
                    [np.expand_dims(self.loaded_original_frames[self.left_frame_index], axis=0),
                     np.expand_dims(self.loaded_original_frames[self.left_frame_index+1], axis=0)])

            gen = gen[0]
            gen = np.squeeze(gen, axis=-1)

            self.interpolated_frame = convert_cv2_image_to_imagepk(256 * gen, flatten=True)

            self.paint_canvas()

            if not self.comparison_mode:
                self.loaded_original_frames = np.concatenate([self.loaded_original_frames[:self.left_frame_index+1],
                                                              np.expand_dims(gen, axis=0),
                                                              self.loaded_original_frames[self.left_frame_index+1:]])

                self.loaded_frames = np.concatenate([self.loaded_frames[:self.left_frame_index + 1],
                                                              np.expand_dims(self.interpolated_frame, axis=0),
                                                              self.loaded_frames[self.left_frame_index + 1:]])


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
        dir_name = filedialog.askdirectory(initialdir="E:\\GianAwesome\\Facultade\\Pesquisa\\Final Phase\\sketch_animated_by_scene",
                                           title="Escolha a pasta com os quadros")

        self.data_loader = DataLoader(dataset_name=os.path.dirname(dir_name),
                                      img_res=(self.frame_network.img_rows, self.frame_network.img_cols),
                                      root_dir=dir_name)

        for scene in sorted(os.listdir(dir_name)):
            first_scene = scene
            break

        self.imgs_A, imgs_B, self.imgs_C = self.data_loader.load_all_image_triplet(first_scene)

        self.loaded_original_frames = imgs_B
        self.loaded_frames = [convert_cv2_image_to_imagepk(256 * img) for img in imgs_B]

        self.paint_canvas()

    def exportar_submenu_on_click(self):
        dir_name = filedialog.askdirectory(initialdir="/",
                                           title="Escolha a pasta para salvar os quadros")

        if dir_name:
            for i, frame in enumerate(self.loaded_original_frames):
                cv2.imwrite(os.path.join(dir_name, str(i).zfill(3) + ".png"), frame * 256)

    def open(self):
        self.window.mainloop()
