import cv2
import PIL
from PIL import ImageTk
import tkinter as tk
from tkinter import font as tkfont
from tkinter import RIGHT, BOTH, RAISED, LEFT, BOTTOM
from gui_dataset import draw_dataset_gui, setup_load_cells, cap


class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(
            family="Helvetica", size=100, weight="bold", slant="italic"
        )
        self.subtitle_font = tkfont.Font(
            family="Helvetica", size=25, weight="bold", slant="italic"
        )
        self.botton_font = tkfont.Font(family="Helvetica", size=14)
        self.page_font = tkfont.Font(family="Helvetica", size=25)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        setup_load_cells()

        self.frames = {}
        for F in (StartPage, program, dataset):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        """Show a frame for the given page name"""
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="PINTO ", font=controller.title_font)
        label2 = tk.Label(
            self, text="Estimate ", font=controller.subtitle_font, pady=20
        )

        # label= tk.Label(self, text="Estimate", font=controller.subtitle_font)
        label.pack()
        label2.pack()

        button1 = tk.Button(
            self,
            text="Estimate",
            command=lambda: controller.show_frame("program"),
            font=controller.botton_font,
            padx=25,
        )
        button2 = tk.Button(
            self,
            text="Make datasets",
            command=lambda: controller.show_frame("dataset"),
            font=controller.botton_font,
        )
        button3 = tk.Button(
            self,
            text="Quit Program",
            command=label.quit,
            font=controller.botton_font,
            bg="red",
            fg="white",
        )

        button1.pack()
        button2.pack()

        button3.pack(side="right")


class program(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(
            self, text="Estimate the marco.... ", font=controller.page_font
        )
        label1 = tk.Label(self, text="ID : ")
        label2 = tk.Label(self, text="NAME : ")
        label3 = tk.Label(self, text="Surname : ")

        button = tk.Button(
            self,
            text="START",
            command=lambda: controller.show_frame("StartPage"),
            font=controller.botton_font,
            bg="green",
            fg="white",
        )
        button2 = tk.Button(
            self,
            text="BACK",
            command=lambda: controller.show_frame("StartPage"),
            font=controller.botton_font,
        )

        imageFrame = tk.Frame(self, width=800, height=600)
        
        
        self.lmain = tk.Label(self)
        self.lmain.grid(row=2)
        
#         self.vid = MyVideoCapture()

#         # Create a canvas that can fit the above video source size
#         self.canvas = tk.Canvas(
#             self, width=self.vid.width, height=self.vid.height
#         )
#         
# 
        label.grid(row=0)
        label1.grid(row=1, column=0)
        tk.Entry(self).grid(row=1, column=1,sticky="W")
        label2.grid(row=2, column=0)
        tk.Entry(self).grid(row=2, column=1,sticky="W")
        label3.grid(row=3, column=0)
        tk.Entry(self).grid(row=3, column=1,sticky="W")
        imageFrame.grid(row=4, column=0, padx=10, pady=2)
#         self.canvas.grid(row=2)
        button.grid(row=100, column=100)
        button2.grid(row=100, column=0)
        
        self.show_frame()

#         self.delay = 15
#         self.update()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame)
            )
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.after(self.delay, self.update)


    def show_frame(self):
        pass
#         _, frame = cap.read()
#         frame = cv2.flip(frame, 1)
#         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
#         img = PIL.Image.fromarray(cv2image)
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.lmain.imgtk = imgtk
#         self.lmain.configure(image=imgtk)
#         self.lmain.after(10, self.show_frame)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        # self.window.mainloop()
        self.mainloop()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)


class dataset(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Dataset", font=controller.title_font)
        label.grid(row=0)
        button = tk.Button(
            self,
            text="Go to the start page",
            command=lambda: controller.show_frame("StartPage"),
        )
        button.grid(row=100)
#         setup_load_cells()
        draw_dataset_gui(self)


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
