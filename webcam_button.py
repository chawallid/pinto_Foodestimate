import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
 

class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source
         
         # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
         
         # create text & Entry 
         self.wanning_label = tkinter.Label(window, text = "***PLEASE INPUT INFORMATION***",fg="red")
         self.wanning_label.grid(row = 0 ,column = 1,sticky="W",pady = 10,padx = 10)
         self.ID_label = tkinter.Label(window, text = "ID :")
         self.ID_label.grid(row = 1 ,column = 0,sticky="W")
         self.ID_entry = tkinter.Entry(window,width = 30)
         self.ID_entry.grid(row = 1 ,column = 1)
 
         self.Name_label = tkinter.Label(window, text = "Name :")
         self.Name_label.grid(row = 2 ,column = 0,sticky="W",pady = 10)
         self.Name_entry = tkinter.Entry(window,width = 30)
         self.Name_entry.grid(row = 2 ,column = 1)

         self.Room_label = tkinter.Label(window, text = "Room :")
         self.Room_label.grid(row = 3 ,column = 0,sticky="W")
         self.Room_entry = tkinter.Entry(window,width = 30)
         self.Room_entry.grid(row = 3 ,column = 1)


         # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
         self.canvas.grid(row = 0 ,column = 2,rowspan = 10,padx=10)
         
         self.lll_label = tkinter.Label(window, text = "weight from sensor (G)")
         self.lll_label.grid(row = 1 ,column = 5)
         self.ll_label = tkinter.Label(window, text = "left weight(G) :")
         self.ll_label.grid(row = 2 ,column = 4)
         self.lt_label = tkinter.Label(window, text = "Top weight(G) :")
         self.lt_label.grid(row = 2 ,column = 5)
         self.lr_label = tkinter.Label(window, text = "Right weight(G) :")
         self.lr_label.grid(row = 2 ,column = 6)

         self.ll_entry = tkinter.Entry(window,width = 10)
         self.ll_entry.grid(row = 3 ,column = 4)
         self.lt_entry = tkinter.Entry(window,width = 10)
         self.lt_entry.grid(row = 3 ,column = 5)
         self.lr_entry = tkinter.Entry(window,width = 10)
         self.lr_entry.grid(row = 3 ,column = 6)

         self.ll_label = tkinter.Label(window, text = "weight from NN (G)")
         self.ll_label.grid(row = 6 ,column = 5)
         self.ll_label = tkinter.Label(window, text = "left weight(G) :")
         self.ll_label.grid(row = 7 ,column = 4)
         self.lt_label = tkinter.Label(window, text = "Top weight(G) :")
         self.lt_label.grid(row = 7 ,column = 5)
         self.lr_label = tkinter.Label(window, text = "Right weight(G) :")
         self.lr_label.grid(row = 7 ,column = 6)

         self.ll_entry = tkinter.Entry(window,width = 10)
         self.ll_entry.grid(row = 8 ,column = 4)
         self.lt_entry = tkinter.Entry(window,width = 10)
         self.lt_entry.grid(row = 8 ,column = 5)
         self.lr_entry = tkinter.Entry(window,width = 10)
         self.lr_entry.grid(row = 8 ,column = 6)

         self.ll_label = tkinter.Label(window, text = "    ")
         self.ll_label.grid(row = 6 ,column = 5,padx  = 10)



         
         # Button that lets the user take a snapshot
         self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot,bg ="green")
         #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
         self.btn_snapshot.grid(row = 11,column = 2)
         self.btn_endt=tkinter.Button(window, text="Quit", width=10, command=window.quit,bg ="red",fg="white")
         self.btn_endt.grid(row=12,column=0)
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 15
         self.update()
 
         self.window.mainloop()
 
     def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             cv2.imwrite("img/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
     def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
 
         if ret:
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
         self.window.after(self.delay, self.update)
 
 
class MyVideoCapture:
     def __init__(self, video_source=0):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
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
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
 
 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Pinto")