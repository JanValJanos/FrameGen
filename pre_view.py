from tkinter import Tk, Button, CENTER
from main_view import MainView


class PreView:
    def __init__(self):
        self.width = 1280
        self.height = 900
        self.window = Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Gerador de quadros intermediários")

        self.add_screen_elements()

    def add_screen_elements(self):
        generation_mode_button = Button(self.window, text="Geração", height=5, width=40,
                                  command=self.generation_mode_button_on_click)
        generation_mode_button.place(relx=0.5, rely=0.4, anchor=CENTER)

        compare_mode_button = Button(self.window, text="Comparação", height=5, width=40,
                                       command=self.compare_mode_button_on_click)
        compare_mode_button.place(relx=0.5, rely=0.6, anchor=CENTER)

    def generation_mode_button_on_click(self):
        main_viewer = MainView(original_window=self.window)
        #main_viewer.open()

    def compare_mode_button_on_click(self):
        main_viewer = MainView(comparison_mode="toggle", original_window=self.window)
        #main_viewer.open()

    def open(self):
        self.window.mainloop()
