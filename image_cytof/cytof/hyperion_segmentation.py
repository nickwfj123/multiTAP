import scipy
import skimage
from skimage import feature
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries

import os
import sys
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # cytof root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from segmentation_functions import generate_mask, normalize

# from cytof.segmentation_functions import generate_mask, normalize


def cytof_nuclei_segmentation(im_nuclei, show_process=False, size_hole=50, size_obj=7,
                              start_coords=(0, 0), side=100, colors=[], min_distance=2,
                              fg_marker_dilate=2, bg_marker_dilate=2
                              ):
    """ Segment nuclei based on the input nuclei image

    Inputs:
        im_nuclei    = raw cytof image correspond to nuclei, size=(h, w)
        show_process = flag of whether show the process  (default=False)
        size_hole    = size of the hole to be removed (default=50)
        size_obj     = size of the small objects to be removed (default=7)
        start_coords = the starting (x,y) coordinates of visualizing process (default=(0,0))
        side         = the side length of visualizing process (default=100)
        colors       = a list of colors used to visualize segmentation results (default=[])
    Returns:
        labels = nuclei segmentation result, where background is represented by 1, size=(h, w)
        colors = the list of colors used to visualize segmentation results

    :param im_nuclei: numpy.ndarray
    :param show_process: bool
    :param size_hole: int
    :param size_obj: int
    :param start_coords: int
    :return labels: numpy.ndarray
    :return colors: list
    """

    if len(colors) == 0:
        cmap_set3 = plt.get_cmap("Set3")
        cmap_tab20c = plt.get_cmap("tab20c")
        colors = [cmap_tab20c.colors[_] for _ in range(len(cmap_tab20c.colors))] + \
                 [cmap_set3.colors[_] for _ in range(len(cmap_set3.colors))]

    x0, y0 = start_coords
    mask = generate_mask(np.clip(im_nuclei, 0, np.quantile(im_nuclei, 0.95)), fill_hole=False, use_watershed=False)
    mask = skimage.morphology.remove_small_holes(mask.astype(bool), size_hole)
    mask = skimage.morphology.remove_small_objects(mask.astype(bool), size_obj)
    if show_process:
        plt.figure(figsize=(4, 4))
        plt.imshow(mask[x0:x0 + side, y0:y0 + side], cmap='gray')
        plt.show()

    # Find and count local maxima
    distance = scipy.ndimage.distance_transform_edt(mask)
    distance = scipy.ndimage.gaussian_filter(distance, 1)
    local_maxi_idx = skimage.feature.peak_local_max(distance, exclude_border=False, min_distance=min_distance,
                                                    labels=None)
    local_maxi = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(local_maxi_idx.T)] = True
    markers = scipy.ndimage.label(local_maxi)[0]
    markers = markers > 0
    markers = skimage.morphology.dilation(markers, skimage.morphology.disk(fg_marker_dilate))
    markers = skimage.morphology.label(markers)
    markers[markers > 0] = markers[markers > 0] + 1
    markers = markers + skimage.morphology.erosion(1 - mask, skimage.morphology.disk(bg_marker_dilate))

    # Another watershed
    temp_im = skimage.util.img_as_ubyte(normalize(np.clip(im_nuclei, 0, np.quantile(im_nuclei, 0.95))))
    gradient = skimage.filters.rank.gradient(temp_im, skimage.morphology.disk(3))
    # gradient = skimage.filters.rank.gradient(normalize(np.clip(im_nuclei, 0, np.quantile(im_nuclei, 0.95))),
    #                                          skimage.morphology.disk(3))
    labels = skimage.segmentation.watershed(gradient, markers)
    labels = skimage.morphology.closing(labels)
    labels_rgb = label2rgb(labels, bg_label=1, colors=colors)
    labels_rgb[labels == 1, ...] = (0, 0, 0)

    if show_process:
        fig, axes = plt.subplots(3, 2, figsize=(8, 12), sharex=False, sharey=False)
        ax = axes.ravel()
        ax[0].set_title("original grayscale")
        ax[0].imshow(np.clip(im_nuclei[x0:x0 + side, y0:y0 + side], 0, np.quantile(im_nuclei, 0.95)),
                     interpolation='nearest')
        ax[1].set_title("markers")
        ax[1].imshow(label2rgb(markers[x0:x0 + side, y0:y0 + side], bg_label=1, colors=colors),
                     interpolation='nearest')
        ax[2].set_title("distance")
        ax[2].imshow(-distance[x0:x0 + side, y0:y0 + side], cmap=plt.cm.nipy_spectral, interpolation='nearest')
        ax[3].set_title("gradient")
        ax[3].imshow(gradient[x0:x0 + side, y0:y0 + side], interpolation='nearest')
        ax[4].set_title("Watershed Labels")
        ax[4].imshow(labels_rgb[x0:x0 + side, y0:y0 + side, :], interpolation='nearest')
        ax[5].set_title("Watershed Labels")
        ax[5].imshow(labels_rgb, interpolation='nearest')
        plt.show()

    return labels, colors


def cytof_cell_segmentation(nuclei_seg, radius=5, membrane_channel=None, show_process=False,
                            start_coords=(0, 0), side=100, colors=[]):
    """ Cell segmentation based on nuclei segmentation; membrane-guided cell segmentation if membrane_channel provided.
    Inputs:
        nuclei_seg       = an index image containing nuclei instance segmentation information, where the background is
                           represented by 1, size=(h,w). Typically, the output of calling the cytof_nuclei_segmentation
                           function.
        radius           = assumed radius of cells (default=5)
        membrane_channel = membrane image channel of original cytof image (default=None)
        show_process     = a flag indicating whether or not showing the segmentation process (default=False)
        start_coords     = the starting (x,y) coordinates of visualizing process (default=(0,0))
        side             = the side length of visualizing process (default=100)
        colors           = a list of colors used to visualize segmentation results (default=[])
    Returns:
        labels = an index image containing cell instance segmentation information, where the background is
                           represented by 1
        colors = the list of colors used to visualize segmentation results

    :param nuclei_seg: numpy.ndarray
    :param radius: int
    :param membrane_channel: numpy.ndarray
    :param show_process: bool
    :param start_coords: tuple
    :param side: int
    :return labels: numpy.ndarray
    :return colors: list
    """

    if len(colors) == 0:
        cmap_set3 = plt.get_cmap("Set3")
        cmap_tab20c = plt.get_cmap("tab20c")
        colors = [cmap_tab20c.colors[_] for _ in range(len(cmap_tab20c.colors))] + \
            [cmap_set3.colors[_] for _ in range(len(cmap_set3.colors))]

    x0, y0 = start_coords

    ## nuclei segmentation -> nuclei mask
    nuclei_mask = nuclei_seg > 1
    if show_process:
        nuclei_bg = nuclei_seg.min()
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        nuclei_seg_vis = label2rgb(nuclei_seg[x0:x0 + side, y0:y0 + side], bg_label=nuclei_bg, colors=colors)
        nuclei_seg_vis[nuclei_seg[x0:x0 + side, y0:y0 + side] == nuclei_bg, ...] = (0, 0, 0)

        ax[0].imshow(nuclei_seg_vis), ax[0].set_title('nuclei segmentation')
        ax[1].imshow(nuclei_mask[x0:x0 + side, y0:y0 + side], cmap='gray'), ax[1].set_title('nuclei mask')

    if membrane_channel is not None:
        membrane_mask = generate_mask(np.clip(membrane_channel, 0, np.quantile(membrane_channel, 0.95)),
                                      fill_hole=False, use_watershed=False)
        if show_process:
            # visualize
            nuclei_membrane = np.zeros((membrane_mask.shape[0], membrane_mask.shape[1], 3), dtype=np.uint8)
            nuclei_membrane[..., 0] = nuclei_mask * 255
            nuclei_membrane[..., 1] = membrane_mask

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(membrane_mask[x0:x0 + side, y0:y0 + side], cmap='gray'), ax[0].set_title('membrane mask')
            ax[1].imshow(nuclei_membrane[x0:x0 + side, y0:y0 + side]), ax[1].set_title('nuclei - membrane')

        # postprocess raw membrane mask
        membrane_mask_close = skimage.morphology.closing(membrane_mask, skimage.morphology.disk(1))
        membrane_mask_open  = skimage.morphology.opening(membrane_mask_close, skimage.morphology.disk(1))
        membrane_mask_erode = skimage.morphology.erosion(membrane_mask_open, skimage.morphology.disk(3))

        # Find skeleton
        membrane_for_skeleton = (membrane_mask_open > 0) & (nuclei_mask == False)
        membrane_skeleton = skimage.morphology.skeletonize(membrane_for_skeleton)
        '''print(membrane_skeleton)
        print(membrane_mask_erode)'''
        membrane_mask = membrane_mask_erode
        membrane_mask_2 = (membrane_mask_erode > 0) | membrane_skeleton

        if show_process:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(membrane_mask[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[0].set_title('raw membrane mask')
            axs[1].imshow(membrane_mask_close[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[1].set_title('membrane mask - closed')
            axs[2].imshow(membrane_mask_open[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[2].set_title('membrane mask -  opened')
            axs[3].imshow(membrane_mask_erode[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[3].set_title('membrane mask - erosion')
            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(membrane_skeleton[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[0].set_title('skeleton')
            axs[1].imshow(membrane_mask[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[1].set_title('membrane mask (final)')
            axs[2].imshow(membrane_mask_2[x0:x0 + side, y0:y0 + side], cmap='gray')
            axs[2].set_title('membrane mask 2')
            plt.show()

            # overlap and visualize
            nuclei_membrane = np.zeros((membrane_mask.shape[0], membrane_mask.shape[1], 3), dtype=np.uint8)
            nuclei_membrane[..., 0] = nuclei_mask * 255
            nuclei_membrane[..., 1] = membrane_mask
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(membrane_mask[x0:x0 + side, y0:y0 + side], cmap='gray'), ax[0].set_title('membrane mask')
            ax[1].imshow(nuclei_membrane[x0:x0 + side, y0:y0 + side]), ax[1].set_title('nuclei - membrane')

    # dilate nuclei mask by radius
    dilate_nuclei_mask = skimage.morphology.dilation(nuclei_mask, skimage.morphology.disk(radius))
    if show_process:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(nuclei_mask[x0:x0 + side, y0:y0 + side], cmap='gray')
        axs[0].set_title('nuclei mask')
        axs[1].imshow(dilate_nuclei_mask[x0:x0 + side, y0:y0 + side], cmap='gray')
        axs[1].set_title('dilated nuclei mask')
        if membrane_channel is not None:
            axs[2].imshow(membrane_mask[x0:x0 + side, y0:y0 + side] > 0, cmap='gray')
            axs[2].set_title('membrane mask')

    # define sure foreground, sure background, and unknown region
    sure_fg = nuclei_mask.copy()  # nuclei mask defines sure foreground

    # dark region in dilated nuclei mask (dilate_nuclei_mask == False) OR bright region in cell mask (cell_mask > 0)
    # defines sure background
    if membrane_channel is not None:
        sure_bg  = ((membrane_mask > 0) | (dilate_nuclei_mask == False)) & (sure_fg == False)
        sure_bg2 = ((membrane_mask_2 > 0) | (dilate_nuclei_mask == False)) & (sure_fg == False)
    else:
        sure_bg =  (dilate_nuclei_mask == False) & (sure_fg == False)

    unknown = np.logical_not(np.logical_or(sure_fg, sure_bg))

    if show_process:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(sure_fg[x0:x0 + side, y0:y0 + side], cmap='gray')
        axs[0].set_title('sure fg')
        axs[1].imshow(sure_bg[x0:x0 + side, y0:y0 + side], cmap='gray')
        if membrane_channel is not None:
            axs[1].set_title('sure bg: membrane | not (dilated nuclei)')
        else:
            axs[1].set_title('sure bg: not (dilated nuclei)')
        axs[2].imshow(unknown[x0:x0 + side, y0:y0 + side], cmap='gray')
        axs[2].set_title('unknown')

        # visualize in a RGB image
        fg_bg_un = np.zeros((unknown.shape[0], unknown.shape[1], 3), dtype=np.uint8)
        fg_bg_un[..., 0] = sure_fg * 255  # sure foreground - red
        fg_bg_un[..., 1] = sure_bg * 255  # sure background - green
        fg_bg_un[..., 2] = unknown * 255  # unknown - blue
        axs[3].imshow(fg_bg_un[x0:x0 + side, y0:y0 + side])
        plt.show()

    ## Euclidean distance transform: distance to the closest zero pixel for each pixel of the input image.
    if membrane_channel is not None:
        distance_bg = -scipy.ndimage.distance_transform_edt(1 - sure_bg2)
        distance_fg = scipy.ndimage.distance_transform_edt(1 - sure_fg)
        distance = distance_bg+distance_fg
    else:
        distance = scipy.ndimage.distance_transform_edt(1 - sure_fg)
    distance = scipy.ndimage.gaussian_filter(distance, 1)

    # watershed
    markers = nuclei_seg.copy()
    markers[unknown] = 0
    if show_process:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].set_title("markers")
        axs[0].imshow(label2rgb(markers[x0:x0 + side, y0:y0 + side], bg_label=1, colors=colors),
                     interpolation='nearest')
        axs[1].set_title("distance")
        im = axs[1].imshow(distance[x0:x0 + side, y0:y0 + side], cmap=plt.cm.nipy_spectral, interpolation='nearest')
        plt.colorbar(im, ax=axs[1])
    labels = skimage.segmentation.watershed(distance, markers)
    if show_process:
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(unknown[x0:x0 + side, y0:y0 + side])
        axs[0].set_title('cytoplasm')  # , cmap=cmap, interpolation='nearest'

        nuclei_lb = label2rgb(nuclei_seg, bg_label=1, colors=colors)
        nuclei_lb[nuclei_seg == 1, ...] = (0, 0, 0)
        axs[1].imshow(nuclei_lb)  # , cmap=cmap, interpolation='nearest')
        axs[1].set_xlim(x0, x0 + side - 1), axs[1].set_ylim(y0 + side - 1, y0)
        axs[1].set_title('nuclei')

        cell_lb = label2rgb(labels, bg_label=1, colors=colors)
        cell_lb[labels == 1, ...] = (0, 0, 0)
        axs[2].imshow(cell_lb)  # , cmap=cmap, interpolation='nearest')
        axs[2].set_title('cells')
        axs[2].set_xlim(x0, x0 + side - 1), axs[2].set_ylim(y0 + side - 1, y0)

        merge_lb = cell_lb.copy()
        merge_lb = cell_lb ** 2
        merge_lb[nuclei_mask == 1, ...] = np.clip(nuclei_lb[nuclei_mask == 1, ...].astype(float) * 1.2, 0, 1)
        axs[3].imshow(merge_lb)
        axs[3].set_title('nuclei-cells')
        axs[3].set_xlim(x0, x0 + side - 1), axs[3].set_ylim(y0 + side - 1, y0)
        plt.show()
    return labels, colors


def visualize_segmentation(raw_image, channels, seg, channel_ids, bound_color=(1, 1, 1), bound_mode='inner', show=True, bg_label=0):

    """ Visualize segmentation results with boundaries
    Inputs:
        raw_image   = raw cytof image
        channels    = a list of channels correspond to each channel in raw_image
        seg         = instance segmentation result (index image)
        channel_ids = indices of desired channels to visualize results
        bound_color = desired color in RGB to show boundaries (default=(1,1,1), white color)
        bound_mode  = the mode for finding boundaries, string in {‘thick’, ‘inner’, ‘outer’, ‘subpixel’}.
                      (default="inner"). For more details, see
                      [skimage.segmentation.mark_boundaries](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
        show        = a flag indicating whether or not print result image on screen
    Returns:
        marked_image
    :param raw_image: numpy.ndarray
    :param seg: numpy.ndarray
    :param channel_ids: int
    :param bound_color: tuple
    :param bound_mode: string
    :param show: bool
    :return marked_image
    """
    from cytof.hyperion_preprocess import cytof_merge_channels

    # mark_boundaries() highight the segmented area for better visualization
    # ref: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
    marked_image = mark_boundaries(cytof_merge_channels(raw_image, channels, channel_ids)[0],
                                   seg, mode=bound_mode, color=bound_color, background_label=bg_label)
    if show:
        plt.figure(figsize=(8,8))
        plt.imshow(marked_image)
        plt.show()
    return marked_image

