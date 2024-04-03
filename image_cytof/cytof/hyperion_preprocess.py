import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import skimage.io as skio
import warnings
from typing import Union, Optional, Type, Tuple, List
# from readimc import MCDFile

# from cytof.classes import CytofImage, CytofImageTiff

import sys
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # cytof root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from classes import CytofImage, CytofImageTiff

# ####################### Read data ########################
def cytof_read_data_roi(filename, slide="", roi=None, iltype="hwd", **kwargs) -> Tuple[CytofImage, list]:
    """ Read cytof data (.txt file) as a dataframe

    Inputs:
        filename = full filename of the cytof data (path-name-ext)

    Returns:
        df_cytof = dataframe of the cytof data
        cols     = column names of the dataframe, an empty list returned if not reading data from a dataframe

    :param filename: str
    :return df_cytof: pandas.core.frame.DataFrame
    """
    ext = pathlib.Path(filename).suffix
    assert len(ext) > 0, "Please provide a full file name with extension!"
    assert ext.upper() in ['.TXT', '.TIFF', '.TIF', '.CSV', '.QPTIFF'], "filetypes other than '.txt', '.tiff'  or '.csv' are not (yet) supported."

    if ext.upper() in ['.TXT', '.CSV']: # the case with a dataframe
        if ext.upper() == '.TXT':
            df_cytof = pd.read_csv(filename, sep='\t') # pd.read_table(filename)
            if roi is None:
                roi = os.path.basename(filename).split('.txt')[0]
            # initialize an instance of CytofImage
            cytof_img = CytofImage(df_cytof, slide=slide, roi=roi, filename=filename)
        elif ext.upper() == '.CSV':
            df_cytof = pd.read_csv(filename) 
            if roi is None:
                roi = os.path.basename(filename).split('.csv')[0]
            # initialize an instance of CytofImage
            cytof_img = CytofImage(df_cytof, slide=slide, roi=roi, filename=filename)
        if "X" in kwargs and "Y" in kwargs:
            cytof_img.df.rename(columns={kwargs["X"]: "X", kwargs["Y"]: 'Y'}, inplace=True)
        cols = cytof_img.df.columns
        

    else: # the case without a dataframe
        image = skio.imread(filename, plugin="tifffile")
        orig_img_shape = image.shape
        sorted_shape = np.sort(orig_img_shape)

        # roll the sorted shape by one to the left 
        # ref: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        correct_shape = np.roll(sorted_shape, -1)

        # sometimes tiff could be square, this ensures images were correctly transposed
        orig_temp = list(orig_img_shape) # tuple is immutable
        correct_index = []
        for shape in correct_shape:
            correct_index.append(orig_temp.index(shape))

            # placeholder, since shape can't = 0
            orig_temp[orig_temp.index(shape)] = 0
        image = image.transpose(correct_index) 

        # create TIFF class cytof image
        cytof_img = CytofImageTiff(image, slide=slide, roi=roi, filename=filename)
        cols = []
          
    return cytof_img, cols

def cytof_read_data_mcd(filename, verbose=False):
    # slides = {}
    cytof_imgs = {}
    with MCDFile(filename) as f:
        if verbose:
            print("\n{}, \n\t{} slides, showing the 1st slide:".format(filename, len(f.slides)))

        ## slide
        for slide in f.slides:
            if verbose:
                print("\tslide ID: {}, description: {}, width: {} um, height: {}um".format(
                slide.id,
                slide.description,
                slide.width_um,
                slide.height_um)
            )
            # slides[slide.id] = {}
            # read the slide image
            im_slide = f.read_slide(slide)  # numpy array or None
            if verbose:
                print("\n\tslide image shape: {}".format(im_slide.shape))

            # (optional) read the first panorama image
            panorama = slide.panoramas[0]
            if verbose:
                print(
                "\t{} panoramas, showing the 1st one. \n\tpanorama ID: {}, description: {}, width: {} um, height: {}um".format(
                    len(slide.panoramas),
                    panorama.id,
                    panorama.description,
                    panorama.width_um,
                    panorama.height_um)
            )
            im_pano = f.read_panorama(panorama)  # numpy array
            if verbose:
                print("\n\tpanorama image shape: {}".format(im_pano.shape))

            for roi in slide.acquisitions: # for each acquisition (roi)
                im_roi = f.read_acquisition(roi)  # array, shape: (c, y, x), dtype: float32
                if verbose:
                    print("\troi {}, shape: {}".format(roi.id, img_roi.shape))
#                 slides[slide.id][roi.id] = {
#                     "channel_names": roi.channel_names,
#                     "channel_labels": roi.channel_labels,
#                     "image": im_roi
#                 }
                cytof_img = CytofImageTiff(image=im_roi.transpose((1,2,0)),
                                           slide=slide.id,
                                           roi=roi.id,
                                           filename=raw_f)
                cytof_img.set_channels(roi.channel_names, roi.channel_labels)
                cytof_imgs["{}_{}".format(slide.id, roi.id)] = cytof_img
    return cytof_imgs# slides


def cytof_preprocess(df):
    """ Preprocess cytof dataframe
        Every pair of X and Y values represent for a unique physical pixel locations in the original image
        The values for Xs and Ys should be continuous integers
        The missing pixels would be filled with 0

    Inputs:
        df = cytof dataframe
        
    Returns:
        df = preprocessed cytof dataframe with missing pixel values filled with 0
        
    :param df: pandas.core.frame.DataFrame
    :return df: pandas.core.frame.DataFrame
    """
    nrow = max(df['Y'].values) + 1
    ncol = max(df['X'].values) + 1
    n = len(df)
    if nrow * ncol > n:
        df2 = pd.DataFrame(np.zeros((nrow * ncol - n, len(df.columns)), dtype=int), columns=df.columns)
        df  = pd.concat([df, df2])
    return df


def cytof_check_channels(df, marker_names=None, xlim=None, ylim=None):
    """A visualization function to show different markers of a cytof image
    
    Inputs:
        df           = preprocessed cytof dataframe
        marker_names = marker names to visualize, should match to column names in df (default=None)
        xlim         = x-axis limit of output image (default=None)
        ylim         = y-axis limit of output image (default=None) 
    
    :param df: pandas.core.frame.DataFrame
    :param marker_names: list
    :param xlim: tuple
    :prarm ylim: tuple
    """
    if marker_names is None:
        marker_names = [df.columns[_] for _ in range(6, len(df.columns))]
    nrow = max(df['Y'].values) + 1
    ncol = max(df['X'].values) + 1
    ax_ncol = 5
    ax_nrow = int(np.ceil(len(marker_names)/5))
    fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3*ax_ncol, 3*ax_nrow))
    if ax_nrow == 1:
        axes = np.array([axes])
    for i, _ in enumerate(marker_names):
        _ax_nrow = int(np.floor(i/ax_ncol))
        _ax_ncol = i % ax_ncol
        image = df[_].values.reshape(nrow, ncol)
        image = np.clip(image/np.quantile(image, 0.99), 0, 1)
        axes[_ax_nrow, _ax_ncol].set_title(_)
        if xlim is not None:
            image = image[:, xlim[0]:xlim[1]]
        if ylim is not None:
            image = image[ylim[0]:ylim[1], :]
        im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
        fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
    plt.show()


def remove_special_channels(self, channels):
    for channel in channels:
        idx = self.channels.index(channel)
        self.channels.pop(idx)
        self.markers.pop(idx)
        self.labels.pop(idx)
        self.df.drop(columns=channel, inplace=True)

def define_special_channels(self, channels_dict):
    # create a copy of original dataframe
    self.df_orig = self.df.copy()
    for new_name, old_names in channels_dict.items():
        print(new_name)
        if len(old_names) == 0:
            continue
        old_nms = []
        for i, old_name in enumerate(old_names):
            if old_name['marker_name'] not in self.channels:
                warnings.warn('{} is not available!'.format(old_name['marker_name']))
                continue
            old_nms.append(old_name)
        if len(old_nms) > 0:
            for i, old_name in enumerate(old_nms):
                if i == 0:
                    self.df[new_name] = self.df[old_name['marker_name']]
                else:
                    self.df[new_name] += self.df[old_name['marker_name']] 
            if not old_name['to_keep']:
                idx = self.channels.index(old_name['marker_name'])
                # Remove the unwanted channels
                self.channels.pop(idx)
                self.markers.pop(idx)
                self.labels.pop(idx)
                self.df.drop(columns=old_name['marker_name'], inplace=True)
            self.channels.append(new_name)

    
def cytof_txt2img(df, marker_names):
    """ Convert from cytof dataframe to d-dimensional image, where d=length of marker names
        Each channel of the output image correspond to the pixel intensity of the corresponding marker
    
    Inputs:
        df           = cytof dataframe
        marker_names = markers to take into consideration
    
    Returns:
        out_img      = d-dimensional image
        
    :param df: pandas.core.frame.DataFrame
    :param marker_names: list
    :return out_img: numpy.ndarray
    """
    nc_in = len(marker_names)
    marker_names = [_ for _ in marker_names if _ in df.columns.values]
    nc = len(marker_names)
    if nc != nc_in:
        warnings.warn("{} markers selected instead of {}".format(nc, nc_in))
    nrow = max(df['Y'].values) + 1
    ncol = max(df['X'].values) + 1
    print("Output image shape: [{}, {}, {}]".format(nrow, ncol, nc))
    out_image = np.zeros([nrow, ncol, nc], dtype=float)
    for _nc in range(nc):
        out_image[..., _nc] = df[marker_names[_nc]].values.reshape(nrow, ncol)
    return out_image


def cytof_merge_channels(im_cytof: np.ndarray,
                         channel_names: List,
                         channel_ids:List = None,
                         channels: List = None,
                         quantiles: List = None,
                         visualize: bool = False):
    """ Merge selected channels (given by "channel_ids") of raw cytof image and generate a RGB image

    Inputs:
        im_cytof      = raw cytof image
        channel_names = a list of names correspond to all channels of the im_cytof
        channel_ids   = the indices of channels to show, no more than 6 channels can be shown the same time (default=None)
        channels      = the names of channels to show, no more than 6 channels can be shown the same time (default=None)
                        Either "channel_ids" or "channels" should be provided
        quantiles     = the quantile values for each channel defined by channel_ids (default=None)
        visualize     = a flag indicating whether print the visualization on screen

    Returns:
        merged_im   = channel merged image
        quantiles   = the quantile values for each channel defined by channel_ids

    :param im_cytof: numpy.ndarray
    :param channel_names: list
    :param channel_ids: list
    :param channels: list
    :param quantiles: list
    :return merged_im: numpy.ndarray
    :return quantiles: list
    """

    assert len(channel_names) == im_cytof.shape[-1], 'The length of "channel_names" does not match the image size!'
    assert channel_ids or channels, 'At least one should be provided, either "channel_ids" or "channels"!'
    if channel_ids is None:
        channel_ids = [channel_names.index(n) for n in channels]
    assert len(channel_ids) <= 6, "No more than 6 channels can be visualized simultaneously!"
    if len(channel_ids) > 3:
        warnings.warn(
            "Visualizing more than 3 channels the same time results in deteriorated visualization. \
            It is not recommended!")

    full_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

    info = [f"{marker} in {c}\n" for (marker, c) in \
            zip([channel_names[i] for i in channel_ids], full_colors[:len(channel_ids)])]
    print(f"Visualizing... \n{''.join(info)}")
    merged_im = np.zeros((im_cytof.shape[0], im_cytof.shape[1], 3))
    if quantiles is None:
        quantiles = [np.quantile(im_cytof[..., _], 0.99) for _ in channel_ids]

    for _ in range(min(len(channel_ids), 3)):
        merged_im[..., _] = np.clip(im_cytof[..., channel_ids[_]] / quantiles[_], 0, 1) * 255

    chs = [[1, 2], [0, 2], [0, 1]]
    chs_id = 0
    while _ < len(channel_ids) - 1:
        _ += 1
        for j in chs[chs_id]:
            merged_im[..., j] += np.clip(im_cytof[..., channel_ids[_]] / quantiles[_], 0, 1) * 255  # /2
            merged_im[..., j] = np.clip(merged_im[..., j], 0, 255)
        chs_id += 1
    merged_im = merged_im.astype(np.uint8)
    if visualize:
        plt.imshow(merged_im)
        plt.show()
    return merged_im, quantiles



