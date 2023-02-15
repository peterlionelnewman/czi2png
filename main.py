#! ~/miniforge3/envs/stl/bin/czi

"""
Written by: Peter Lionel Harry Newmman, 2023, (p.newman @ sydney edu au)

Helpful for students
1. GUI interface to:
2. search a folder for .czi files
3. export mips of various kinds for each czi channel

Go check out the Allen Institue of Cell Science package !!!

This exists to help students with mip generation, and because of the bugs in the mosaic builder

"""

from sys import platform
import os
import re
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageChops
import multiprocessing
import time
import numpy as np
import attr
import warnings
from scipy.ndimage import zoom
from tkinter.filedialog import askdirectory
import tkinter as tk
import aicspylibczi
from matplotlib import pyplot as plt


class GUI(tk.Tk):
    """

    Create a tkinter window to select options:

    - select a folder to search for czi files
    - select check boxes for:
        - side projections (w/ side project scaling)
        - gamma correction
        - save mip channels
        - save mip panel
        - save mip merge
        - save dye overlaid
        - save colors
        - use multiprocessing
    - select a folder to save the images to

    """

    def __init__(self):
        # create a tkinter window
        self.root = tk.Tk()
        self.root.title(' ')

        # initalize variables
        self.search_path = tk.StringVar()
        self.search_path.set('.set/search/path')
        self.save_path = tk.StringVar()
        self.save_path.set('.set/save/path')

        # establish a scale factor
        s = 1.0

        w = int(420 * s)
        h = int(230 * s)

        # size the window
        self.root.minsize(w, h)
        self.root.maxsize(w, h)
        self.root.geometry(f'{w}x{h}')

        # add a title
        tk.Label(self.root, text='czi2png', font=('Arial', 25))\
            .place(relx=10 / w, rely=10 / h) # width=135/w, height=36/h)

        # # add search and save path buttons
        tk.Button(self.root, text='Search Path', command=self.specify_search_path,
                  width=8, height=1).place(relx=10 / w, rely=53 / h)
        tk.Button(self.root, text='Save Path', command=self.specify_save_path,
                  width=8, height=1).place(relx=10 / w, rely=83 / h)

        self.search_path_label = tk.Label(self.root, textvariable=self.search_path, font=('Arial', 12), fg='gray')\
            .place(relx=124 / w, rely=58 / h) # width=135/w, height=36/h)
        self.save_path_label = tk.Label(self.root, textvariable=self.save_path, font=('Arial', 12), fg='gray') \
            .place(relx=124 / w, rely=88 / h)  # width=135/w, height=36/h)

        # # add a button to run the program
        tk.Button(self.root, text='Convert 2 png!', command=self.main,
                  width=20, height=2).place(relx=10 / w, rely=140 / h)

        # # add checkbox to save mip channels, mip panel, mip merge, dye overlaid, colors, multiprocessing
        tk.Label(self.root, text='Options', font=('Arial', 12)) \
            .place(relx=260 / w, rely=22 / h)  # width=135/w, height=36/h)
        self.save_mip_channels = tk.BooleanVar()
        self.save_mip_channels.set(False)
        tk.Checkbutton(self.root, text='save mip channels', command = self.display_input,
                       variable=self.save_mip_channels, onvalue=1, offvalue=0)\
            .place(relx=260 / w, rely=53 / h)

        self.save_mip_panel = tk.BooleanVar()
        self.save_mip_panel.set(False)
        tk.Checkbutton(self.root, text='save mip panel', command = self.display_input,
                       variable=self.save_mip_panel, onvalue=1, offvalue=0) \
            .place(relx=260 / w, rely=80 / h)

        self.save_mip_merge = tk.BooleanVar()
        self.save_mip_merge.set(False)
        tk.Checkbutton(self.root, text='save mip merge', command = self.display_input,
                       variable=self.save_mip_merge, onvalue=1, offvalue=0) \
            .place(relx=260 / w, rely=107 / h)

        self.save_dye_overlaid = tk.BooleanVar()
        self.save_dye_overlaid.set(False)
        tk.Checkbutton(self.root, text='save dye overlaid', command = self.display_input,
                       variable=self.save_dye_overlaid, onvalue=1, offvalue=0) \
            .place(relx=260 / w, rely=133 / h)

        self.save_colors = tk.BooleanVar()
        self.save_colors.set(False)
        tk.Checkbutton(self.root, text='save colors', command = self.display_input,
                       variable=self.save_colors, onvalue=1, offvalue=0) \
            .place(relx=260 / w, rely=160 / h)

        self.use_multiprocessing = tk.BooleanVar()
        self.use_multiprocessing.set(False)
        tk.Checkbutton(self.root, text='use multiprocessing', command = self.display_input,
                       variable=self.use_multiprocessing, onvalue=1, offvalue=0) \
            .place(relx=260 / w, rely=186 / h)


        self.root.mainloop()

    # debugging
    def display_input(self):
        print(f'search path: {self.search_path}')
        print(f'save path: {self.save_path}')
        print(f'save mip channels: {self.save_mip_channels.get()}')
        print(f'save mip panel: {self.save_mip_panel.get()}')
        print(f'save mip merge: {self.save_mip_merge.get()}')
        print(f'save dye overlaid: {self.save_dye_overlaid.get()}')
        print(f'save colors: {self.save_colors.get()}')
        print(f'use multiprocessing: {self.use_multiprocessing.get()}')


    def specify_search_path(self,):
        self.search_path.set(tk.filedialog.askdirectory(parent=self.root, initialdir='/',
                                        title='Please select a directory'))

    def specify_save_path(self,):
        self.save_path.set(tk.filedialog.askdirectory(parent=self.root, initialdir='/',
                                        title='Please select a directory'))


    def main(self):
        if self.search_path.get() == '.set/search/path' or self.save_path.get() == '.set/save/path':
            tk.messagebox.showerror('Python Error', 'please select Search and Save paths * unassigned *')
            return

        if not os.path.isdir(self.search_path.get()) or not os.path.isdir(self.save_path.get()):
            tk.messagebox.showerror('Python Error', 'Search and Save paths not directories')
            return

        # check that at least one save option is selected
        if self.save_mip_channels.get() + self.save_mip_panel.get() + self.save_mip_merge.get() < 1:
            tk.messagebox.showerror('Python Error', 'select at least channels, panel or merge image to save')
            return

        czi_files = find_all_czi_in_path(self.search_path.get())
        # check that there are czi files in the search path
        if len(czi_files) == 0:
            print('no czi files found in search path')
            return

        czi_file_sizes = [os.path.getsize(czi_file) for czi_file in czi_files]
        cumulative_file_size = np.cumsum(czi_file_sizes)

        # zip each czi file with processing parameters
        process_params = zip(czi_files,
                             [self.save_path.get()] * len(czi_files),
                             [self.save_mip_channels.get()] * len(czi_files),
                             [self.save_mip_panel.get()] * len(czi_files),
                             [self.save_mip_merge.get()] * len(czi_files),
                             [self.save_dye_overlaid.get()] * len(czi_files),
                             [self.save_colors.get()] * len(czi_files))

        start_time = time.perf_counter()
        # run the processing routine on the czi images
        if not self.use_multiprocessing.get():
            print(f'running single threaded on: {czi_files}')
            # time the function
            # run the processing routine on the czi images
            for n, process_param in enumerate(process_params):
                process_file(process_param)

                # calculate time remaining use the files size to estimate processing time
                time_remaining = (time.perf_counter() - start_time) / \
                                 cumulative_file_size[n] \
                                 * (cumulative_file_size[-1] - cumulative_file_size[n])

                                # time taken
                                # per byte
                                # * bytes remaining

                print(f'Time remaining: {time_remaining:0.2f} seconds')

        else:
            print(f'running multi processed: {czi_files}')
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.map(process_file, process_params)

        # print the run time
        print(f'\nTime elapsed: {time.perf_counter() - start_time:0.2f} seconds')

        # display a message box to indicate that the processing is complete
        tk.messagebox.showinfo('Python Info', 'Images saved as png')


def find_all_czi_in_path(search_path):
    """
    search a given directory for all czis
    """
    os.chdir(search_path)
    czi_files = []
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith('.czi') or file.endswith('.CZI'):
                czi_files.append(os.path.join(root, file))

    return czi_files


@attr.s(auto_attribs=True, auto_detect=True)
class CziImage:
    """
    czi image class for lite image processing

    wrapper class around the ACIS wrapper class;
    around the czilib library
    """

    # _: list = attr.ib(default=[' '])
    # _: str = attr.ib(default='')
    # _: np.ndarray = attr.ib(default=[0., 0.])
    # _: int = attr.ib(default=0)

    path: str = attr.ib(default='')
    czifile: aicspylibczi.CziFile = attr.ib(default=None)
    czi: tuple = attr.ib(default=())
    im: np.ndarray = attr.ib(default=[0., 0.])
    num_channels: int = attr.ib(default=0)
    num_z_slices: int = attr.ib(default=0)
    height: int = attr.ib(default=0)
    width: int = attr.ib(default=0)
    num_timepoints: int = attr.ib(default=0)
    num_scenes: int = attr.ib(default=0)
    num_blocks: int = attr.ib(default=0)
    num_mosaics: int = attr.ib(default=0)
    metadata: str = attr.ib(default='')
    mip: np.ndarray = attr.ib(default=[0., 0.])
    colours: list = attr.ib(default=[])
    dyes: list = attr.ib(default=[])

    def load_czi(self, path):
        """
        load a czi file
        """
        # set the path
        self.path = path

        # get the image
        self.czifile = aicspylibczi.CziFile(self.path)
        dims = self.czifile.get_dims_shape()[0]

        if dims['X'][1] == 0 and dims['Y'][1] == 0:
            return 'metadata_only'

        # get the number of channels, z_slices and shape
        self.num_channels = dims['C'][1]
        self.num_z_slices = dims['Z'][1]
        self.width = dims['X'][1]
        self.height = dims['Y'][1]

        # load each image
        if self.czifile.is_mosaic():

            # additional checks
            if not 'M' in dims:
                raise ValueError('Mosaic image found, but no M dimension found')

            # load in all the bounding boxes
            self.num_mosaics = dims['M'][1]
            bbox = np.zeros((self.num_mosaics, 4)).astype(int)
            for m in range(self.num_mosaics):
                _ = self.czifile.get_mosaic_tile_bounding_box(C=0, Z=0, M=m)
                bbox[m, :] = int(_.x), int(_.y), int(_.w), int(_.h)

            # check that w and h is the same
            if not np.all(bbox[:, 2] == bbox[0, 2]) or not np.all(bbox[:, 3] == bbox[0, 3]):
                raise ValueError('Mosaic bounding boxes are not the same size')

            # simplify the box
            bbox[:, 0] = bbox[:, 0] - np.min(bbox[:, 0])
            bbox[:, 1] = bbox[:, 1] - np.min(bbox[:, 1])

            # initialize the image
            self.width = np.max(bbox[:, 0]) + bbox[0, 2]
            self.height = np.max(bbox[:, 1]) + bbox[0, 3]
            # some rounding bug means i need to add 1 to the width and height
            self.im = np.zeros((self.num_channels, self.num_z_slices, self.height + 1, self.width + 1))

            # load the image
            self.czi = self.czifile.read_image()

            # get the 'im' == (czyx)
            self.temp = np.moveaxis(self.czi[0],
                                    [self.czifile.dims.index('C'),
                                     self.czifile.dims.index('Z'),
                                     self.czifile.dims.index('Y'),
                                     self.czifile.dims.index('X'),
                                     self.czifile.dims.index('M')],
                                    [0, 1, 2, 3, 4])

            # index the last dimensions at 0
            for i in range(len(dims) - 5):
                self.temp = self.temp[..., 0]

            # move the mosaics into im
            for m in range(self.num_mosaics):
                self.im[:, :, bbox[m, 1]:bbox[m, 1] + bbox[m, 3], bbox[m, 0]:bbox[m, 0] + bbox[m, 2]] = self.temp[:, :, :, :, m]

        else:
            self.czi = self.czifile.read_image()

            # get the 'im' == (czyx)
            self.im = np.moveaxis(self.czi[0],
                                  [self.czifile.dims.index('C'),
                                   self.czifile.dims.index('Z'),
                                   self.czifile.dims.index('Y'),
                                   self.czifile.dims.index('X')],
                                  [0, 1, 2, 3])

            # index the last dimensions at 0
            for i in range(len(dims) - 4):
                self.im = self.im[..., 0]

        # convert im to f64
        self.im = self.im.astype(np.float64)

        # get the other info:
        if 'T' in dims: # timepoints
            # self.num_timepoints = dims['T'][1]
            # warnings.warn('this script throws away this info')
            pass
        if 'S' in dims: # scenes
            # self.num_scenes = dims['S'][1]
            # warnings.warn('this script throws away this info')
            pass
        if 'B' in dims: # blocks
            # self.num_blocks = dims['B'][1]
            # warnings.warn('this script throws away this info')
            pass
        if 'V' in dims:
            # The V-dimension ('view').
            # warnings.warn(''V' dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'I' in dims:
            # The I-dimension ('illumination').
            # warnings.warn(''I' dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'R' in dims:
            # The R-dimension ('rotation').
            # warnings.warn(''R' dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'H' in dims:
            # The H-dimension ('phase').
            # warnings.warn(''H' dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass


    def extract_colors(self):
        self.metadata = self.czifile.meta
        dyes = self.metadata.findall('.//DyeName')

        if len(dyes) != self.num_channels:
            warnings.warn('num channel != num dyes')
            return

        self.dyes = [None] * self.num_channels

        for c in range(self.num_channels):
            self.dyes[c] = dyes[c].text

            if self.dyes[c] == 'DAPI'\
                    or self.dyes[c] == 'dapi'\
                    or self.dyes[c] == 'Hoechst 33342'\
                    or self.dyes[c] == 'Hoechst 33258':
                self.colours.append([0, 255, 255])
                continue
            elif self.dyes[c] == 'FITC':
                self.colours.append([255, 255, 0])
                continue
            elif self.dyes[c] == 'Cy3':
                self.colours.append([255, 0, 0])
                continue
            elif self.dyes[c] == 'Cy5':
                self.colours.append([255, 0, 255])
                continue

            # extract all numbers from dye
            dye_nums = float(re.findall(r'\d+', self.dyes[c])[0])

            if dye_nums < 405:
                self.colours.append([0, 255, 255])
            elif dye_nums < 500:
                self.colours.append([255, 255, 0])
            elif dye_nums < 600:
                self.colours.append([255, 0, 0])
            elif dye_nums < 700:
                self.colours.append([255, 0, 255])
            else:
                self.colours.append([0, 0, 0])
                warnings.warn(f'no color found for {self.path}; channel {c}, {self.dyes[c]}')


    def project_mip(self, side_projections=False, z_scale=1):
        """
        make a maximum intensity projection
        """
        # check for z slices
        if self.num_z_slices == 1:
            warnings.warn(f'no z slices found in {self.path} returning')
            self.mip = np.zeros((self.num_channels,
                                 self.im.shape[2],
                                 self.im.shape[3]))
            for c in range(self.num_channels):
                self.mip[c, :, :] = self.im[c, 0, :, :]
            return

        # initialize the mip
        if side_projections:
            self.mip = np.zeros((self.num_channels,
                                 self.im.shape[2] + self.im.shape[1] * z_scale + 1,
                                 self.im.shape[3] + self.im.shape[1] * z_scale + 1))

            for c in range(self.num_channels):
                # check for z slices
                self.mip[c,
                        0:self.im.shape[2],
                        0:self.im.shape[3]] = \
                        np.max(self.im[c, :, :, :], axis=0)

                projection_yz = np.max(self.im[c, :, :, :], axis=1)
                projection_yz = zoom(projection_yz, (z_scale, 1))
                self.mip[c,
                        (self.im.shape[2] + 1):(self.im.shape[2] + 1 + self.im.shape[1] * z_scale),
                        0:self.im.shape[3]] = \
                        projection_yz

                projection_xz = np.max(self.im[c, :, :, :], axis=2).T
                projection_xz = zoom(projection_xz, (1, z_scale))
                self.mip[c,
                        0:self.im.shape[2],
                        (self.im.shape[3] + 1):(self.im.shape[3] + 1 + self.im.shape[1]  * z_scale)] = \
                        projection_xz

        else:
            self.mip = np.zeros((self.num_channels,
                                 self.im.shape[2],
                                 self.im.shape[3]))

            for c in range(self.num_channels):
                self.mip[c, :, :] = np.max(self.im[c, :, :, :], axis=0)


    def normalize(self, max=False, gamma=1):
        """
        modifies '.mip' normalizing the image
        """
        for c in range(self.num_channels):

            if max:
                mip = (self.mip[c, :, :] - np.nanmin(self.mip[c, :, :])) / \
                                      (np.nanmax(self.mip[c, :, :]) - np.nanmin(self.mip[c, :, :]))
            else:
                mip = self.mip[c, :, :] - np.nanmin(self.mip[c, :, :])
                if np.nansum(mip) != 0:
                    mip_p = np.percentile(mip, 99.8)
                    mip[mip > mip_p] = mip_p
                    mip = mip/mip_p

            if gamma != 1:
                mip = mip ** gamma

            self.mip[c, :, :] = mip * 255


    def save(self,
             save_path,
             save_mip_channels,
             save_mip_panel,
             save_mip_merge,
             save_dye_overlaid,
             save_colors):

        cwd = os.getcwd()
        os.chdir(save_path)

        file_stem = os.path.splitext(os.path.basename(self.path))[0]

        for c in range(self.num_channels):
            # PIL on each channel
            mip = Image.fromarray(self.mip[c, :, :])
            mip = mip.convert('L')
            base = np.ceil(self.num_channels ** 0.5).astype('int')

            if save_colors and len(self.colours[c]) > 0:
                mip = ImageOps.colorize(mip, (0, 0, 0), tuple(self.colours[c]))
            else:
                mip = mip.convert('RGB')

            if save_dye_overlaid:
                font_color = tuple(self.colours[c])
                font_size = self.height // 50
                draw = ImageDraw.Draw(mip)
                if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
                    text_overlay = [[]]
                    text_overlay.append(c * ['\n'])
                    text_overlay.append([self.dyes[c]])
                    text_overlay = ''.join([item for sublist in text_overlay for item in sublist])
                    draw.text((0, 0), text_overlay,
                              font_color,
                              ImageFont.truetype('Arial.ttf', size=font_size))
                elif platform == 'win32':
                    draw.text((0, 0), self.dyes[c],
                              font_color,
                              ImageFont.truetype('arial.ttf', size=font_size))

            # create an image of all 'c' merged
            if c == 0:
                mip_merge = mip.copy()
            else:
                mip_merge = Image.merge('RGB', (
                    ImageChops.add(mip_merge.getchannel('R'), mip.getchannel('R')),
                    ImageChops.add(mip_merge.getchannel('G'), mip.getchannel('G')),
                    ImageChops.add(mip_merge.getchannel('B'), mip.getchannel('B'))))

            if self.num_channels > 1:
                if c == 0:
                    mip_panel = np.zeros((mip.height * base, mip.width * base, 3))

                mip_panel[c % base * mip.height:(c % base + 1) * mip.height,
                          c // base * mip.width:(c // base + 1) * mip.width,
                          :] \
                            = np.array(mip).copy()

            # save
            if save_mip_channels:
                mip.save(f'{file_stem[:-4]}_ch{c}_.png', optimize=True)

        # save merged and panel images
        if save_mip_merge:
            mip_merge.save(f'{file_stem[:-4]}_ch{c}_merge.png', optimize=True)

        if save_mip_panel and self.num_channels > 1:
            # add the merge to the panel if there is space
            if self.num_channels < base ** 2:
                mip_panel[self.num_channels % base * mip.height:(self.num_channels % base + 1) * mip.height,
                          self.num_channels // base * mip.width:(self.num_channels // base + 1) * mip.width,
                          :] \
                            = np.array(mip_merge).copy()

            # remove black space
            mip_panel = mip_panel[~np.all(mip_panel == 0, axis=(1, 2))]
            mip_panel = Image.fromarray(mip_panel.astype('uint8'))
            mip_panel.save(f'{file_stem[:-4]}_ch{c}_panel.png', optimize=True)

        print(f'converted {file_stem} and saved')
        os.chdir(cwd)


def process_file(process_params):
    """
    processes to run on an image path
    """

    path,\
    save_path,\
    save_mip_channels,\
    save_mip_panel,\
    save_mip_merge,\
    save_dye_overlaid,\
    save_colors = process_params

    print(f'processing path: {path}\n'
          f'with save path: {save_path}; '
          f'and save options: \n'
          f'-   channels: {save_mip_channels}\n'
          f'-   panel: {save_mip_panel}\n'
          f'-   merge: {save_mip_merge}\n'
          f'-   dye overlay: {save_dye_overlaid}\n'
          f'-   colors: {save_colors}\n')


    czi = CziImage() # initiate
    # process the image
    if czi.load_czi(path) == 'metadata_only':
        return
    czi.extract_colors()
    czi.project_mip(side_projections=True, z_scale=3)
    czi.normalize(gamma=1)
    czi.save(save_path,
             save_mip_channels,
             save_mip_panel,
             save_mip_merge,
             save_dye_overlaid,
             save_colors)


if __name__ == '__main__':
    gui = GUI()


