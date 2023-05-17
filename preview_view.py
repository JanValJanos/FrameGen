from tkinter import Tk, Button, CENTER, Label, Frame, Toplevel
import time

class PreviewView(Toplevel):
    def __init__(self, loaded_frames, master=None):
        self.width = 512
        self.height = 552

        self.loaded_frames = loaded_frames
        self.cur_frame = 0

        super().__init__(master=master)

        self.geometry(f"{self.width}x{self.height}")
        self.title("Inbetween frame generator")

        play_button = Button(self, text="Play", height=5, width=20,
                                  command=self.show_animation)
        play_button.place(relx=0.5, rely=0.80, anchor=CENTER)

    def show_animation(self):
        if self.loaded_frames is not None:
            for i in range(len(self.loaded_frames)):
                self.after(80 * i, self.show_frame)

    def show_frame(self):
        print(self.cur_frame)
        self.place_frame_label(self.loaded_frames[self.cur_frame], relx=0.5, rely=0.4)
        if self.cur_frame >= len(self.loaded_frames) - 1:
            self.cur_frame = 0
        else:
            self.cur_frame += 1

    def place_frame_label(self, loaded_frame, relx=0, rely=0, anchor=CENTER):
        label = self.create_frame_label(loaded_frame)
        label.place(relx=relx, rely=rely, anchor=anchor)

        Label(self, text=("I-" if loaded_frame.interpolated else "") + str(loaded_frame.name)).place(relx=relx, rely=rely+0.27, anchor=anchor)

    def create_frame_label(self, loaded_frame):
        color = "#000000"
        border_frame = Frame(self, background=color)

        label = Label(border_frame, image=loaded_frame.frame)
        label.pack(padx=3, pady=3)

        return border_frame

    def open(self):
        self.show_animation()