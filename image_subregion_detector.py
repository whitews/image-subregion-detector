import Tkinter
import tkFileDialog
from PIL import ImageTk
import PIL.Image
import os
import cv2
import numpy as np

from isd_lib import utils

BACKGROUND_COLOR = '#ededed'

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 680

PAD_SMALL = 2
PAD_MEDIUM = 4
PAD_LARGE = 8
PAD_EXTRA_LARGE = 14

DEFAULT_DILATE_ITER = 3


class Application(Tkinter.Frame):

    def __init__(self, master):

        Tkinter.Frame.__init__(self, master=master)

        self.image_name = None
        self.image_dir = None

        self.master.minsize(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        self.main_frame = Tkinter.Frame(self.master, bg=BACKGROUND_COLOR)
        self.main_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            padx=0,
            pady=0
        )

        self.left_frame = Tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.left_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            side=Tkinter.LEFT,
            padx=0,
            pady=0
        )

        self.right_frame = Tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.right_frame.pack(
            fill=Tkinter.Y,
            expand=False,
            side=Tkinter.LEFT,
            padx=PAD_MEDIUM,
            pady=40
        )

        file_chooser_frame = Tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        file_chooser_frame.pack(
            fill=Tkinter.X,
            expand=False,
            anchor=Tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_chooser_button = Tkinter.Button(
            file_chooser_frame,
            text='Choose Image File...',
            command=self.choose_files
        )
        file_chooser_button.pack(side=Tkinter.LEFT)

        # the canvas frame's contents will use grid b/c of the double
        # scrollbar (they don't look right using pack), but the canvas itself
        # will be packed in its frame
        canvas_frame = Tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.pack(
            fill=Tkinter.BOTH,
            expand=True,
            anchor=Tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        self.canvas = Tkinter.Canvas(canvas_frame, cursor="cross")

        self.scrollbar_v = Tkinter.Scrollbar(
            canvas_frame,
            orient=Tkinter.VERTICAL
        )
        self.scrollbar_h = Tkinter.Scrollbar(
            canvas_frame,
            orient=Tkinter.HORIZONTAL
        )
        self.scrollbar_v.config(command=self.canvas.yview)
        self.scrollbar_h.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.scrollbar_v.set)
        self.canvas.config(xscrollcommand=self.scrollbar_h.set)

        self.canvas.grid(
            row=0,
            column=0,
            sticky=Tkinter.N + Tkinter.S + Tkinter.E + Tkinter.W
        )
        self.scrollbar_v.grid(row=0, column=1, sticky=Tkinter.N + Tkinter.S)
        self.scrollbar_h.grid(row=1, column=0, sticky=Tkinter.E + Tkinter.W)

        # start packing in right_frame
        dilate_frame = Tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        dilate_frame.pack(
            fill=Tkinter.BOTH,
            expand=False,
            anchor=Tkinter.N,
            pady=PAD_MEDIUM
        )
        self.dilate_iter = Tkinter.IntVar()
        self.dilate_iter.set(3)
        dilate_label = Tkinter.Label(
            dilate_frame,
            text="Dilation iterations: ",
            bg=BACKGROUND_COLOR
        )
        dilate_label_entry = Tkinter.Entry(
            dilate_frame,
            textvariable=self.dilate_iter
        )
        dilate_label_entry.config(width=4)
        dilate_label_entry.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)
        dilate_label.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)

        min_area_frame = Tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        min_area_frame.pack(
            fill=Tkinter.BOTH,
            expand=False,
            anchor=Tkinter.N,
            pady=PAD_MEDIUM
        )
        self.min_area = Tkinter.DoubleVar()
        self.min_area.set(0.5)
        min_area_label = Tkinter.Label(
            min_area_frame,
            text="Minimum area: ",
            bg=BACKGROUND_COLOR
        )
        min_area_label_entry = Tkinter.Entry(
            min_area_frame,
            textvariable=self.min_area
        )
        min_area_label_entry.config(width=4)
        min_area_label_entry.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)
        min_area_label.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)
        
        max_area_frame = Tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        max_area_frame.pack(
            fill=Tkinter.BOTH,
            expand=False,
            anchor=Tkinter.N,
            pady=PAD_MEDIUM
        )
        self.max_area = Tkinter.DoubleVar()
        self.max_area.set(2.0)
        max_area_label = Tkinter.Label(
            max_area_frame,
            text="Minimum area: ",
            bg=BACKGROUND_COLOR
        )
        max_area_label_entry = Tkinter.Entry(
            max_area_frame,
            textvariable=self.max_area
        )
        max_area_label_entry.config(width=4)
        max_area_label_entry.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)
        max_area_label.pack(side=Tkinter.RIGHT, anchor=Tkinter.N)

        find_regions_button = Tkinter.Button(
            self.right_frame,
            text='Find Regions',
            command=self.find_regions
        )
        find_regions_button.pack(side=Tkinter.LEFT, anchor=Tkinter.N)

        # setup some button and key bindings
        self.canvas.bind("<ButtonPress-1>", self.on_draw_button_press)
        self.canvas.bind("<B1-Motion>", self.on_draw_move)

        self.canvas.bind("<ButtonPress-2>", self.on_pan_button_press)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.pan_start_x = None
        self.pan_start_y = None

        self.image = None
        self.tk_image = None

        self.pack()

    def on_draw_button_press(self, event):
        # starting coordinates
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create a new rectangle if we don't already have one
        if self.rect is None:
            self.rect = self.canvas.create_rectangle(
                self.start_x,
                self.start_y,
                self.start_x,
                self.start_y,
                outline='green',
                width=2
            )

    def on_draw_move(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # update rectangle size with mouse position
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_pan_button_press(self, event):
        self.canvas.config(cursor='fleur')

        # starting position for panning
        self.pan_start_x = int(self.canvas.canvasx(event.x))
        self.pan_start_y = int(self.canvas.canvasy(event.y))

    def pan_image(self, event):
        self.canvas.scan_dragto(
            event.x - self.pan_start_x,
            event.y - self.pan_start_y,
            gain=1
        )

    # noinspection PyUnusedLocal
    def on_pan_button_release(self, event):
        self.canvas.config(cursor='cross')

    def find_regions(self):
        corners = self.canvas.coords(self.rect)
        corners = tuple([int(c) for c in corners])
        region = self.image.crop(corners)

        hsv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)
        target = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2HSV)

        region_mask, rectangles = utils.find_regions(
            hsv_img,
            target,
            dilate=self.dilate_iter.get(),
            min_area=self.min_area.get(),
            max_area=self.max_area.get()
        )

        self.draw_rectangles(rectangles)

    def draw_rectangles(self, rectangles):
        for rect in rectangles:
            self.canvas.create_rectangle(
                rect[0],
                rect[1],
                rect[0] + rect[2],
                rect[1] + rect[3],
                outline='green',
                width=2
            )

        self.canvas.delete(self.rect)
        self.rect = None

    def choose_files(self):
        self.canvas.delete(self.rect)
        self.rect = None

        selected_file = tkFileDialog.askopenfile('r')

        self.image = PIL.Image.open(selected_file)
        height, width = self.image.size
        self.canvas.config(scrollregion=(0, 0, height, width))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=Tkinter.NW, image=self.tk_image)

        self.image_name = os.path.basename(selected_file.name)
        self.image_dir = os.path.dirname(selected_file.name)

root = Tkinter.Tk()
app = Application(root)
root.mainloop()
