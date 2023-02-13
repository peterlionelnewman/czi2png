#! ~/miniforge3/envs/stl/bin/czi

"""
Written by: Pete, 2023, (peterlionelnewman @ gmail com / p.newman @ sydney edu au)

Helpful for students
1. searches a folder for .czi files
2. exports a mip of each czi channel

Go check out the Allen Institue of Cell Science package !!!

This exists because sometimes its buggy. But otherwise we AOK.

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
import aicspylibczi
from matplotlib import pyplot as plt


def find_all_czi_in_path(master_path):
    """
    search a given directory for images
    """
    os.chdir(master_path)
    czi_files = []
    for root, _, files in os.walk(master_path):
        for file in files:
            if file.endswith(".czi") or file.endswith(".CZI"):
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
            # The V-dimension ("view").
            # warnings.warn('"V" dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'I' in dims:
            # The I-dimension ("illumination").
            # warnings.warn('"I" dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'R' in dims:
            # The R-dimension ("rotation").
            # warnings.warn('"R" dimension found, not sure what this is')
            # warnings.warn('this script throws away this info')
            pass
        if 'H' in dims:
            # The H-dimension ("phase").
            # warnings.warn('"H" dimension found, not sure what this is')
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
             save_mip_channels=False,
             save_mip_panel=True,
             save_mip_merge=True,
             save_dye_overlaid=True,
             save_colors=True, ):

        for c in range(self.num_channels):
            # PIL on each channel
            mip = Image.fromarray(self.mip[c, :, :])
            mip = mip.convert('L')
            base = np.ceil(self.num_channels ** 0.5).astype('int')

            if save_colors and len(self.colours[c]) > 0:
                mip = ImageOps.colorize(mip, (0, 0, 0), tuple(self.colours[c]))

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
                mip.save(f'{self.path[:-4]}_ch{c}_.png', optimize=True)

        # save merged and panel images
        if save_mip_merge:
            mip_merge.save(f'{self.path[:-4]}_ch{c}_merge.png', optimize=True)

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
            mip_panel.save(f'{self.path[:-4]}_ch{c}_panel.png', optimize=True)

        print(f'converted {self.path} and saved')


def process_file(path):
    """
    processes to run on an image path
    """

    print(f'processing {path}')

    czi = CziImage() # initiate
    # process the image
    if czi.load_czi(path) == 'metadata_only':
        return
    czi.extract_colors()
    czi.project_mip(side_projections=True, z_scale=3)
    czi.normalize(gamma=1)
    czi.save(save_mip_channels=False,
             save_mip_panel=True,
             save_mip_merge=True,
             save_dye_overlaid=True,
             save_colors=True)


if __name__ == '__main__':

    # time the function
    start_time = time.perf_counter()

    # add path to search for czis
    master_path = '/Users/peternewman/Desktop/test_images/New-05-1.czi'
    master_path = askdirectory()

    czi_files = find_all_czi_in_path(master_path)
    czi_file_sizes = [os.path.getsize(czi_file) for czi_file in czi_files]
    cumulative_file_size = np.cumsum(czi_file_sizes)


    multi_processed = True
    # run the processing routine on the czi images
    if not multi_processed:
        # run the processing routine on the czi images
        for n, czi_file in enumerate(czi_files):
            process_file(czi_file)

            # calculate time remaining use the files size to estimate processing time
            time_remaining = (time.perf_counter() - start_time) / \
                             cumulative_file_size[n] \
                             * (cumulative_file_size[-1] - cumulative_file_size[n])

                            # time taken
                            # per byte
                            # * bytes remaining

            print(f'Time remaining: {time_remaining:0.2f} seconds')

    else:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.map(process_file, czi_files)

    # print the run time
    print(f'\nTime elapsed: {time.perf_counter() - start_time:0.2f} seconds')

    print('TBD map all colors to one mega image')