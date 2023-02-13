from tkinter import Tk, Button, Canvas, Menu, CENTER
from FrameNetwork import FrameNetwork


class MainView:
    def __init__(self):
        self.frame_network = None
        self.width = 1280
        self.height = 720
        self.window = Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Gerador de quadros intermediários")

        self.paint_canvas()

        self.add_screen_elements()
        self.add_toolbar()

        self.build_network()

    def paint_canvas(self):
        canvas = Canvas(self.window, bg='lightgray', height=self.height, width=self.width)
        canvas.place(relx=0, rely=0)

        canvas.create_rectangle(100, 150, 405, 300)
        canvas.create_rectangle(482, 300, 797, 450)
        canvas.create_rectangle(865, 150, 1180, 300)

    def add_screen_elements(self):
        add_frame_button = Button(self.window, text="Gerar quadro", height=5, width=40,
                                  command=self.add_frame_button_on_click)
        add_frame_button.place(relx=0.5, rely=0.73, anchor=CENTER)

        add_all_frames_button = Button(self.window, text="Gerar quadros para todos os pares", height=5, width=40,
                                       command=self.add_all_frames_button_on_click)
        add_all_frames_button.place(relx=0.5, rely=0.85, anchor=CENTER)

        move_left_button = Button(self.window, text="<", height=5, width=5,
                                  command=self.move_left_button_on_click)
        move_left_button.place(relx=0.3, rely=0.80, anchor=CENTER)

        move_right_button = Button(self.window, text=">", height=5, width=5,
                                   command=self.move_right_button_on_click)
        move_right_button.place(relx=0.7, rely=0.80, anchor=CENTER)

    def add_toolbar(self):
        main_toolbar = Menu(self.window)
        self.window.config(menu=main_toolbar)

        importar_submenu = Menu(main_toolbar)
        main_toolbar.add_cascade(label="Importar", menu=importar_submenu)
        importar_submenu.add_command(label="Arquivos", command=self.importar_submenu_on_click)

    def build_network(self):
        self.frame_network = FrameNetwork()

        self.frame_network.load_weights("E:\\GianAwesome\\Facultade\\outside_drive\\douga-keras-pix2pix-anime-optimizes-testbed-2400-atd12k-3.h5")

    def add_frame_button_on_click(self):
        self.frame_network.generator.predict()

    def add_all_frames_button_on_click(self):
        pass

    def move_left_button_on_click(self):
        pass

    def move_right_button_on_click(self):
        pass

    def importar_submenu_on_click(self):
        pass

    def open(self):
        self.window.mainloop()
