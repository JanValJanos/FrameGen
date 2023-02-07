from tkinter import Tk, Button, Canvas, CENTER


class MainView:
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.window = Tk()
        self.window.geometry(f"{self.width}x{self.height}")
        self.window.title("Gerador de quadros intermediários")

        self.paint_canvas()

        add_frame_button = Button(self.window, text="Gerar quadro", height=5, width=40,
                                  command=self.add_frame_button_on_click)
        add_frame_button.place(relx=0.5, rely=0.73, anchor=CENTER)

        add_frame_button = Button(self.window, text="Gerar quadros para todos os pares", height=5, width=40,
                                  command=self.add_frame_button_on_click)
        add_frame_button.place(relx=0.5, rely=0.85, anchor=CENTER)

        add_frame_button = Button(self.window, text="<", height=5, width=5,
                                  command=self.add_frame_button_on_click)
        add_frame_button.place(relx=0.3, rely=0.80, anchor=CENTER)

        add_frame_button = Button(self.window, text=">", height=5, width=5,
                                  command=self.add_frame_button_on_click)
        add_frame_button.place(relx=0.7, rely=0.80, anchor=CENTER)

    def paint_canvas(self):
        canvas = Canvas(self.window, bg='lightgray', height=self.height, width=self.width)
        canvas.place(relx=0, rely=0)

        canvas.create_rectangle(100, 150, 405, 300)
        canvas.create_rectangle(482, 300, 797, 450)
        canvas.create_rectangle(865, 150, 1180, 300)

    def add_frame_button_on_click(self):
        pass

    def open(self):
        self.window.mainloop()
