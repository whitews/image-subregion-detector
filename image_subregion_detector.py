import tkinter
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk
import PIL.Image
import os
import re
import cv2
import numpy as np

from isd_lib import utils

BACKGROUND_COLOR = '#ededed'

WINDOW_WIDTH = 990
WINDOW_HEIGHT = 720

PREVIEW_SIZE = 256  # height & width of preview in pixels

PAD_SMALL = 2
PAD_MEDIUM = 4
PAD_LARGE = 8
PAD_EXTRA_LARGE = 14

DEFAULT_ERODE_ITER = 0
DEFAULT_DILATE_ITER = 2

COLOR_NAMES = [
    'red',
    'yellow',
    'green',
    'cyan',
    'blue',
    'violet',
    'white',
    'gray',
    'black'
]


class Application(tkinter.Frame):

    def __init__(self, master):

        tkinter.Frame.__init__(self, master=master)

        self.image_name = None
        self.image_dir = None
        self.bg_colors = None

        # Detected regions will be saved as a dictionary with the bounding
        # rectangles canvas ID as the key. The value will be another dictionary
        # containing the contour itself and the rectangle coordinates
        self.regions = None

        self.region_count = tkinter.IntVar()
        self.region_min = tkinter.DoubleVar()
        self.region_max = tkinter.DoubleVar()
        self.region_avg = tkinter.DoubleVar()

        self.master.minsize(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self.master.title("Image Sub-region Detector")

        self.main_frame = tkinter.Frame(self.master, bg=BACKGROUND_COLOR)
        self.main_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            padx=0,
            pady=0
        )

        self.left_frame = tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.left_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            side=tkinter.LEFT,
            padx=0,
            pady=0
        )

        self.right_frame = tkinter.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.right_frame.pack(
            fill=tkinter.Y,
            expand=False,
            side=tkinter.LEFT,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_chooser_frame = tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        file_chooser_frame.pack(
            fill=tkinter.X,
            expand=False,
            anchor=tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        file_chooser_button = tkinter.Button(
            file_chooser_frame,
            text='Choose Image File...',
            command=self.choose_files
        )
        file_chooser_button.pack(side=tkinter.LEFT)

        self.export_format = tkinter.StringVar()
        self.export_format.set('numpy')
        format_label = tkinter.Label(
            file_chooser_frame,
            text="  Export Format: ",
            bg=BACKGROUND_COLOR
        )
        export_fmt_combo = tkinter.OptionMenu(
            file_chooser_frame,
            self.export_format,
            "numpy",
            "tiff",
            "both"
        )
        export_fmt_combo.config(width=6)
        export_fmt_combo.pack(side=tkinter.RIGHT)
        format_label.pack(side=tkinter.RIGHT)

        self.export_string = tkinter.StringVar()
        snip_label = tkinter.Label(
            file_chooser_frame,
            text="Export Label: ",
            bg=BACKGROUND_COLOR
        )
        snip_label_entry = tkinter.Entry(
            file_chooser_frame,
            textvariable=self.export_string
        )
        snip_label_entry.pack(side=tkinter.RIGHT)
        snip_label.pack(side=tkinter.RIGHT)

        # the canvas frame's contents will use grid b/c of the double
        # scrollbar (they don't look right using pack), but the canvas itself
        # will be packed in its frame
        canvas_frame = tkinter.Frame(self.left_frame, bg=BACKGROUND_COLOR)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            anchor=tkinter.N,
            padx=PAD_MEDIUM,
            pady=PAD_MEDIUM
        )

        self.canvas = tkinter.Canvas(canvas_frame, cursor="cross")

        self.scrollbar_v = tkinter.Scrollbar(
            canvas_frame,
            orient=tkinter.VERTICAL
        )
        self.scrollbar_h = tkinter.Scrollbar(
            canvas_frame,
            orient=tkinter.HORIZONTAL
        )
        self.scrollbar_v.config(command=self.canvas.yview)
        self.scrollbar_h.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.scrollbar_v.set)
        self.canvas.config(xscrollcommand=self.scrollbar_h.set)

        self.canvas.grid(
            row=0,
            column=0,
            sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W
        )
        self.scrollbar_v.grid(row=0, column=1, sticky=tkinter.N + tkinter.S)
        self.scrollbar_h.grid(row=1, column=0, sticky=tkinter.E + tkinter.W)

        # start packing in right_frame
        export_button = tkinter.Button(
            self.right_frame,
            text='Export Sub-regions',
            command=self.export_sub_regions
        )
        export_button.pack(fill=tkinter.BOTH, anchor=tkinter.N)

        bg_colors_frame = tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        bg_colors_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=(PAD_EXTRA_LARGE, PAD_MEDIUM)
        )
        bg_colors_label = tkinter.Label(
            bg_colors_frame,
            text="Background colors: ",
            bg=BACKGROUND_COLOR
        )
        bg_colors_label.pack(side=tkinter.TOP, anchor=tkinter.W)

        color_profile_frame = tkinter.Frame(
            bg_colors_frame,
            bg=BACKGROUND_COLOR
        )
        color_profile_frame.pack(
            fill=tkinter.X,
            expand=True,
            anchor=tkinter.W,
            side=tkinter.LEFT
        )

        self.color_profile_vars = {}
        for color in COLOR_NAMES:
            self.color_profile_vars[color] = tkinter.StringVar()
            self.color_profile_vars[color].set("0.0%")
            l = tkinter.Label(
                color_profile_frame,
                textvariable=self.color_profile_vars[color],
                bg=BACKGROUND_COLOR
            )
            l.config(
                borderwidth=0,
                highlightthickness=0
            )
            l.pack(anchor=tkinter.E, pady=PAD_SMALL, padx=PAD_MEDIUM)

        bg_cb_frame = tkinter.Frame(bg_colors_frame, bg=BACKGROUND_COLOR)
        bg_cb_frame.pack(
            fill=tkinter.NONE,
            expand=False,
            anchor=tkinter.E
        )

        self.bg_color_vars = {}
        for color in COLOR_NAMES:
            self.bg_color_vars[color] = tkinter.IntVar()
            self.bg_color_vars[color].set(0)
            cb = tkinter.Checkbutton(
                bg_cb_frame,
                text=color,
                variable=self.bg_color_vars[color],
                bg=BACKGROUND_COLOR
            )
            cb.config(
                borderwidth=0,
                highlightthickness=0
            )
            cb.pack(anchor=tkinter.W, pady=PAD_SMALL, padx=PAD_MEDIUM)

        erode_frame = tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        erode_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_MEDIUM
        )
        self.erode_iter = tkinter.IntVar()
        self.erode_iter.set(DEFAULT_ERODE_ITER)
        erode_label = tkinter.Label(
            erode_frame,
            text="Erosion iterations: ",
            bg=BACKGROUND_COLOR
        )
        erode_label_entry = tkinter.Entry(
            erode_frame,
            textvariable=self.erode_iter
        )
        erode_label_entry.config(width=4)
        erode_label_entry.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        erode_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)

        dilate_frame = tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        dilate_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_MEDIUM
        )
        self.dilate_iter = tkinter.IntVar()
        self.dilate_iter.set(DEFAULT_DILATE_ITER)
        dilate_label = tkinter.Label(
            dilate_frame,
            text="Dilation iterations: ",
            bg=BACKGROUND_COLOR
        )
        dilate_label_entry = tkinter.Entry(
            dilate_frame,
            textvariable=self.dilate_iter
        )
        dilate_label_entry.config(width=4)
        dilate_label_entry.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        dilate_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)

        min_area_frame = tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        min_area_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_MEDIUM
        )
        self.min_area = tkinter.DoubleVar()
        self.min_area.set(0.5)
        min_area_label = tkinter.Label(
            min_area_frame,
            text="Minimum area: ",
            bg=BACKGROUND_COLOR
        )
        min_area_label_entry = tkinter.Entry(
            min_area_frame,
            textvariable=self.min_area
        )
        min_area_label_entry.config(width=4)
        min_area_label_entry.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        min_area_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        
        max_area_frame = tkinter.Frame(self.right_frame, bg=BACKGROUND_COLOR)
        max_area_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_MEDIUM
        )
        self.max_area = tkinter.DoubleVar()
        self.max_area.set(2.0)
        max_area_label = tkinter.Label(
            max_area_frame,
            text="Maximum area: ",
            bg=BACKGROUND_COLOR
        )
        max_area_label_entry = tkinter.Entry(
            max_area_frame,
            textvariable=self.max_area
        )
        max_area_label_entry.config(width=4)
        max_area_label_entry.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        max_area_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)

        region_buttons_frame = tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR
        )
        region_buttons_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_MEDIUM
        )

        find_regions_button = tkinter.Button(
            region_buttons_frame,
            text='Find Regions',
            command=self.find_regions
        )
        find_regions_button.pack(side=tkinter.LEFT, anchor=tkinter.N)

        clear_regions_button = tkinter.Button(
            region_buttons_frame,
            text='Clear Regions',
            command=self.clear_rectangles
        )
        clear_regions_button.pack(side=tkinter.LEFT, anchor=tkinter.N)

        # frame showing various stats about found regions
        stats_frame = tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR,
            highlightthickness=1,
            highlightbackground='gray'
        )
        stats_frame.pack(
            fill=tkinter.BOTH,
            expand=False,
            anchor=tkinter.N,
            pady=PAD_LARGE,
            padx=PAD_MEDIUM
        )
        region_count_frame = tkinter.Frame(
            stats_frame,
            bg=BACKGROUND_COLOR
        )
        region_count_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            anchor=tkinter.N,
            pady=PAD_SMALL,
            padx=PAD_SMALL
        )
        region_count_desc_label = tkinter.Label(
            region_count_frame,
            text="# of regions: ",
            bg=BACKGROUND_COLOR
        )
        region_count_desc_label.pack(side=tkinter.LEFT, anchor=tkinter.N)
        region_count_label = tkinter.Label(
            region_count_frame,
            textvariable=self.region_count,
            bg=BACKGROUND_COLOR
        )
        region_count_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        
        region_min_frame = tkinter.Frame(
            stats_frame,
            bg=BACKGROUND_COLOR
        )
        region_min_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            anchor=tkinter.N,
            pady=PAD_SMALL,
            padx=PAD_SMALL
        )
        region_min_desc_label = tkinter.Label(
            region_min_frame,
            text="Minimum size: ",
            bg=BACKGROUND_COLOR
        )
        region_min_desc_label.pack(side=tkinter.LEFT, anchor=tkinter.N)
        region_min_label = tkinter.Label(
            region_min_frame,
            textvariable=self.region_min,
            bg=BACKGROUND_COLOR
        )
        region_min_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        
        region_max_frame = tkinter.Frame(
            stats_frame,
            bg=BACKGROUND_COLOR
        )
        region_max_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            anchor=tkinter.N,
            pady=PAD_SMALL,
            padx=PAD_SMALL
        )
        region_max_desc_label = tkinter.Label(
            region_max_frame,
            text="Maximum size: ",
            bg=BACKGROUND_COLOR
        )
        region_max_desc_label.pack(side=tkinter.LEFT, anchor=tkinter.N)
        region_max_label = tkinter.Label(
            region_max_frame,
            textvariable=self.region_max,
            bg=BACKGROUND_COLOR
        )
        region_max_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)
        
        region_avg_frame = tkinter.Frame(
            stats_frame,
            bg=BACKGROUND_COLOR
        )
        region_avg_frame.pack(
            fill=tkinter.BOTH,
            expand=True,
            anchor=tkinter.N,
            pady=PAD_SMALL,
            padx=PAD_SMALL
        )
        region_avg_desc_label = tkinter.Label(
            region_avg_frame,
            text="Average size: ",
            bg=BACKGROUND_COLOR
        )
        region_avg_desc_label.pack(side=tkinter.LEFT, anchor=tkinter.N)
        region_avg_label = tkinter.Label(
            region_avg_frame,
            textvariable=self.region_avg,
            bg=BACKGROUND_COLOR
        )
        region_avg_label.pack(side=tkinter.RIGHT, anchor=tkinter.N)

        # preview frame holding small full-size depiction of chosen image
        preview_frame = tkinter.Frame(
            self.right_frame,
            bg=BACKGROUND_COLOR,
            highlightthickness=1,
            highlightbackground='black'
        )
        preview_frame.pack(
            fill=tkinter.NONE,
            expand=False,
            anchor=tkinter.S,
            side=tkinter.BOTTOM
        )

        self.preview_canvas = tkinter.Canvas(
            preview_frame,
            highlightthickness=0
        )
        self.preview_canvas.config(width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        self.preview_canvas.pack(anchor=tkinter.S, side=tkinter.BOTTOM)

        # setup some button and key bindings
        self.canvas.bind("<ButtonPress-1>", self.on_draw_button_press)
        self.canvas.bind("<B1-Motion>", self.on_draw_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_draw_release)

        self.canvas.bind("<ButtonPress-2>", self.on_pan_button_press)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_button_release)

        self.canvas.bind("<ButtonPress-3>", self.on_right_button_press)

        self.canvas.bind("<Configure>", self.canvas_size_changed)

        self.scrollbar_h.bind("<B1-Motion>", self.update_preview)
        self.scrollbar_h.bind("<ButtonRelease-1>", self.update_preview)
        self.scrollbar_v.bind("<B1-Motion>", self.update_preview)
        self.scrollbar_v.bind("<ButtonRelease-1>", self.update_preview)

        self.preview_canvas.bind("<ButtonPress-1>", self.move_preview_rectangle)
        self.preview_canvas.bind("<B1-Motion>", self.move_preview_rectangle)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.pan_start_x = None
        self.pan_start_y = None

        self.image = None
        self.tk_image = None
        self.preview_image = None
        self.preview_rectangle = None

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
                outline='#00ff00',
                width=2
            )

    def on_draw_move(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # update rectangle size with mouse position
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    # noinspection PyUnusedLocal
    def on_draw_release(self, event):
        if self.rect is None or self.image is None:
            return

        corners = self.canvas.coords(self.rect)
        corners = tuple([int(c) for c in corners])
        region = self.image.crop(corners)

        if 0 in region.size:
            # either height or width is zero, do nothing
            return

        target = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2HSV)

        color_profile = utils.get_color_profile(target)

        total_pixels = (corners[2] - corners[0]) * (corners[3] - corners[1])

        for color in COLOR_NAMES:
            color_percent = (float(color_profile[color]) / total_pixels) * 100
            self.color_profile_vars[color].set(
                "%.1f%%" % np.round(color_percent, decimals=1)
            )

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
        self.update_preview(None)

    # noinspection PyUnusedLocal
    def on_pan_button_release(self, event):
        self.canvas.config(cursor='cross')

    def on_right_button_press(self, event):
        # have to translate our event position to our current panned location
        selection = self.canvas.find_closest(
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
            start='rect'
        )

        for item in selection:
            tags = self.canvas.gettags(item)

            if 'rect' not in tags:
                # this isn't a rectangle object, do nothing
                continue

            self.canvas.delete(item)
            self.regions.pop(item)

    def find_regions(self):
        if self.rect is None or self.image is None:
            return

        corners = self.canvas.coords(self.rect)
        corners = tuple([int(c) for c in corners])
        region = self.image.crop(corners)

        hsv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)
        target = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2HSV)

        bg_colors = []
        for color, cb_var in self.bg_color_vars.items():
            if cb_var.get() == 1:
                bg_colors.append(color)

        if len(bg_colors) <= 0:
            messagebox.showwarning(
                'Choose Background Color',
                'Please choose at least one background color to find regions.'
            )
            return

        contours = utils.find_regions(
            hsv_img,
            target,
            bg_colors=bg_colors,
            pre_erode=self.erode_iter.get(),
            dilate=self.dilate_iter.get(),
            min_area=self.min_area.get(),
            max_area=self.max_area.get()
        )

        # make sure we have at least one detected region
        if len(contours) > 0:
            self.create_regions(contours)
        else:
            self.region_count.set(0)
            self.region_min.set(0.0)
            self.region_max.set(0.0)
            self.region_avg.set(0.0)

    def create_regions(self, contours):
        """
        Creates regions (self.regions) & draws bounding rectangles on canvas

        Args:
            contours: list of OpenCV contours
        """
        self.clear_rectangles()
        self.regions = {}  # reset regions dictionary

        region_areas = []

        for c in contours:
            region_areas.append(cv2.contourArea(c))
            rect = cv2.boundingRect(c)

            # using a custom fully transparent bitmap for the stipple, b/c
            # if the rectangle has no fill we cannot catch mouse clicks
            # within its boundaries (only on the border itself)
            # a bit of a hack but it works
            rect_id = self.canvas.create_rectangle(
                rect[0],
                rect[1],
                rect[0] + rect[2],
                rect[1] + rect[3],
                outline='#00ff00',
                fill='gray',
                stipple='@trans.xbm',
                width=2,
                tag='rect'
            )

            self.regions[rect_id] = {
                'contour': c,
                'rectangle': rect
            }

        self.region_count.set(len(contours))
        self.region_min.set(min(region_areas))
        self.region_max.set(max(region_areas))
        self.region_avg.set(np.round(np.mean(region_areas), decimals=1))

    def reset_color_profile(self):
        for color in COLOR_NAMES:
            self.color_profile_vars[color].set("0.0%")

    def clear_rectangles(self):
        self.canvas.delete("rect")
        self.canvas.delete(self.rect)
        self.rect = None
        self.region_count.set(0)
        self.region_min.set(0.0)
        self.region_max.set(0.0)
        self.region_avg.set(0.0)
        self.reset_color_profile()

    def set_preview_rectangle(self):
        x1, x2 = self.scrollbar_h.get()
        y1, y2 = self.scrollbar_v.get()

        self.preview_rectangle = self.preview_canvas.create_rectangle(
            int(x1 * PREVIEW_SIZE) + 1,
            int(y1 * PREVIEW_SIZE) + 1,
            int(x2 * PREVIEW_SIZE),
            int(y2 * PREVIEW_SIZE),
            outline='#00ff00',
            width=2,
            tag='preview_rect'
        )

    # noinspection PyUnusedLocal
    def update_preview(self, event):
        if self.preview_rectangle is None:
            # do nothing
            return

        x1, x2 = self.scrollbar_h.get()
        y1, y2 = self.scrollbar_v.get()

        # current rectangle position
        rx1, ry1, rx2, ry2 = self.preview_canvas.coords(
            self.preview_rectangle
        )

        delta_x = int(x1 * PREVIEW_SIZE) + 1 - rx1
        delta_y = int(y1 * PREVIEW_SIZE) + 1 - ry1

        self.preview_canvas.move(
            self.preview_rectangle,
            delta_x,
            delta_y
        )

    def move_preview_rectangle(self, event):
        if self.preview_rectangle is None:
            # do nothing
            return

        x1, y1, x2, y2 = self.preview_canvas.coords(self.preview_rectangle)

        half_width = float(x2 - x1) / 2
        half_height = float(y2 - y1) / 2

        if event.x + half_width >= PREVIEW_SIZE - 1:
            new_x = PREVIEW_SIZE - (half_width * 2) - 1
        else:
            new_x = event.x - half_width

        if event.y + half_height >= PREVIEW_SIZE - 1:
            new_y = PREVIEW_SIZE - (half_height * 2) - 1
        else:
            new_y = event.y - half_height

        self.canvas.xview(
            tkinter.MOVETO,
            float(new_x) / PREVIEW_SIZE
        )

        self.canvas.yview(
            tkinter.MOVETO,
            float(new_y) / PREVIEW_SIZE
        )

        self.update()
        self.update_preview(None)

    # noinspection PyUnusedLocal
    def canvas_size_changed(self, event):
        self.preview_canvas.delete('preview_rect')
        self.set_preview_rectangle()

    def choose_files(self):
        selected_file = filedialog.askopenfile('r')

        if selected_file is None:
            # do nothing, user cancelled file dialog
            return

        self.canvas.delete('all')
        self.rect = None
        self.region_count.set(0)
        self.region_min.set(0.0)
        self.region_max.set(0.0)
        self.region_avg.set(0.0)

        # some of the files may be 3-channel 16-bit/chan TIFFs, which
        # PIL doesn't support. OpenCV can read these, but converts them
        # to 8-bit/chan. So, we'll open all images in OpenCV first,
        # then create a PIL Image to finally create an ImageTk PhotoImage
        cv_img = cv2.imread(selected_file.name)

        self.image = PIL.Image.fromarray(
            cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB),
            'RGB'
        )
        height, width = self.image.size
        self.canvas.config(scrollregion=(0, 0, height, width))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tkinter.NW, image=self.tk_image)

        # have to force an update of the UI else the canvas scroll bars
        # will not have updated fast enough to get their positions for
        # drawing the preview rectangle
        self.update()

        tmp_preview_image = self.image.resize(
            (PREVIEW_SIZE, PREVIEW_SIZE),
            PIL.Image.ANTIALIAS
        )
        self.preview_canvas.delete('all')
        self.preview_image = ImageTk.PhotoImage(tmp_preview_image)
        self.preview_canvas.create_image(
            0,
            0,
            anchor=tkinter.NW,
            image=self.preview_image
        )
        self.set_preview_rectangle()

        self.image_name = os.path.basename(selected_file.name)
        self.image_dir = os.path.dirname(selected_file.name)

    def export_sub_regions(self):
        if not self.regions:
            return

        if len(self.regions) == 0:
            return

        if self.export_string.get() == '':
            messagebox.showwarning(
                'Export Label',
                'Please create an export label.'
            )
            return

        export_format = self.export_format.get()

        output_dir = "/".join(
            [
                self.image_dir,
                self.export_string.get().strip()
            ]
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        hsv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)

        for k, v in self.regions.items():
            x1 = v['rectangle'][0]
            y1 = v['rectangle'][1]
            x2 = v['rectangle'][0] + v['rectangle'][2]
            y2 = v['rectangle'][1] + v['rectangle'][3]

            # build base file name for output files
            match = re.search('(.+)\.(.+)$', self.image_name)
            output_filename = "".join(
                [
                    match.groups()[0],
                    '_',
                    str(x1),
                    ',',
                    str(y1)
                ]
            )

            if export_format == 'tiff' or export_format == 'both':
                region = self.image.crop((x1, y1, x2, y2))
                tif_filename = ".".join([output_filename, 'tif'])
                tif_file_path = "/".join([output_dir, tif_filename])
                region.save(tif_file_path)

            if export_format == 'numpy' or export_format == 'both':
                # extract sub-region from original image using rectangle
                hsv_region = hsv_img[y1:y2, x1:x2]

                # subtract the rect coordinates from the contour
                local_contour = v['contour'] - [x1, y1]

                # create a mask from the new contour
                new_mask = np.zeros(
                    (v['rectangle'][3], v['rectangle'][2]),
                    dtype=np.uint8
                )
                cv2.drawContours(new_mask, [local_contour], 0, 255, -1)

                # mask the extracted sub-region & convert to int16
                # to use -1 for non-contour pixels
                masked_region = cv2.bitwise_and(
                    hsv_region,
                    hsv_region,
                    mask=new_mask
                ).astype(np.int16)

                # set non-contour areas to -1
                masked_region[new_mask == 0] = -1

                # save sub-region to file as NumPy array
                npy_filename = ".".join([output_filename, 'npy'])
                npy_file_path = "/".join([output_dir, npy_filename])
                np.save(npy_file_path, masked_region)

root = tkinter.Tk()
app = Application(root)
root.mainloop()
