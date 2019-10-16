import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog
from tkinter import ttk

import matplotlib
import matplotlib.image as mtlib_img
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

matplotlib.use('TKAgg')
LARGE_FONT = ('Verdana', 12)
x1 = y1 = x2 = y2 = 0
objects = {
    'forest': (0, 255, 0),
    'crops': (255, 255, 0),
    'urban': (0, 0, 255)
}


@dataclass
class ObjectType:
    name: str
    rgb_value: tuple
    x1: int
    y1: int
    x2: int
    y2: int


class MainClass(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.container = tk.Frame(self)
        self.container.pack(side='top', fill='both', expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.shared_data = {
            'out_path': tk.StringVar(),
            'in_path': tk.StringVar(),
            'dest_path': tk.StringVar()
        }
        self.geometry('1000x1000')
        self.frames = {}
        self.add_activity(StartActivity)
        self.show_activity(StartActivity)

    def add_activity(self, Activity):
        frame = Activity(self.container, self)
        self.frames[Activity] = frame
        frame.grid(row=0, column=0, sticky='nsew')

    def show_activity(self, controller):
        frame = self.frames[controller]
        frame.tkraise()

    def get_activity(self, activity):
        return self.frames[activity]


class StartActivity(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text='Generate clustering map', font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        first_activity_bttn = ttk.Button(self, text='Load the files',
                                         command=lambda: [self.controller.add_activity(PlotActivity),
                                                          self.controller.show_activity(PlotActivity)])
        first_activity_bttn.pack()
        self.gen_map_button = ttk.Button(self, text="Select output map",
                                         command=lambda: self.file_dialog('out_path'))
        self.gen_map_button.pack()
        self.input_map_button = ttk.Button(self, text="Select input map",
                                           command=lambda: self.file_dialog('in_path'))
        self.input_map_button.pack()
        self.dest_path_button = ttk.Button(self, text="Select dest path",
                                           command=lambda: self.dir_dialog('dest_path'))
        self.dest_path_button.pack()

    def file_dialog(self, key):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetype=[('All files', '*.*')])
        self.controller.shared_data[key].set(self.filename)

    def dir_dialog(self, key):
        self.filename = filedialog.askdirectory(initialdir="/", title="Select directory")
        self.controller.shared_data[key].set(self.filename)


class PlotActivity(tk.Frame):
    object_colors = {
        'forest': 'green',
        'crops': 'yellow',
        'urban': 'blue'
    }

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.groups = []
        self.controller = controller
        button1 = ttk.Button(self, text='Go to start frame',
                             command=lambda: controller.show_activity(StartActivity))
        button1.pack()
        choices = {'Forest', 'Crops', 'Urban'}
        self.menu_choice = tk.StringVar()
        popup_menu = ttk.OptionMenu(self, self.menu_choice, 'Forest', *choices)
        popup_menu.pack()

        generate_button = ttk.Button(self, text='Generate map', command=self.generate_map)
        generate_button.pack()

        input_path = controller.shared_data['in_path'].get()
        output_path = controller.shared_data['out_path'].get()
        self.dest_path = controller.shared_data['dest_path'].get()
        self.out = np.load(output_path).copy()
        self.img = mtlib_img.imread(input_path)
        self.np_img = np.asarray(self.img).copy()
        fig = Figure()
        self.ax = fig.add_subplot()
        self.colors = {'forest': mpatches.Patch(color='green', label='Forest'),
                       'urban': mpatches.Patch(color='blue', label='Urban'),
                       'crops': mpatches.Patch(color='yellow', label='Crops')
                       }
        implot = self.ax.imshow(self.img, aspect='auto')
        canvas = FigureCanvasTkAgg(fig, self)
        toggle_selector.RS = RectangleSelector(self.ax, self.line_select_callback, drawtype='box', useblit=True,
                                               button=[1, 3],
                                               minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                               rectprops=dict(facecolor='white',
                                                              edgecolor='black', alpha=0.1, fill=True))
        canvas.mpl_connect('key_press_event', toggle_selector)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def line_select_callback(self, eclick, erelease):
        global x1, y1, x2, y2
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        print(x1, y1, '\n', x2, y2)
        self.groups.append(ObjectType(self.menu_choice.get().lower(),
                                      objects[self.menu_choice.get().lower()],
                                      x1, y1, x2, y2))
        print(self.groups)

    def generate_map(self):
        colors = []
        for group in self.groups:
            colors.append(self.colors[group.name])
            chunk = self.out[group.y1: group.y2, group.x1: group.x2].ravel()
            counts = np.bincount(chunk)
            argmax = np.argmax(counts)
            to_color = np.where(self.out == argmax)
            for row, col in zip(*to_color):
                self.np_img[row, col] = group.rgb_value
        fig = plt.figure(dpi=100, tight_layout=True, frameon=False,
                         figsize=(self.np_img.shape[0] / 100, self.np_img.shape[1] / 100))
        fig.figimage(self.np_img, cmap=plt.cm.binary)
        fig.legend(handles=colors, fontsize='25', loc='lower right')
        plt.savefig(os.path.join(self.dest_path, 'legend_plot.jpg'))
        Image.fromarray(self.np_img).save(os.path.join(self.dest_path, 'high_res.jpg'))


def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        toggle_selector.RS.set_active(True)


def run():
    app = MainClass()
    app.mainloop()


if __name__ == '__main__':
    run()
