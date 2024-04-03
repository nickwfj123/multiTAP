#!/usr/bin/env python
# coding: utf-8
import os
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import argparse
import yaml
import pandas as pd
import skimage

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


# import sys
# sys.path.append('../cytof')
from hyperion_preprocess import cytof_read_data_roi
from hyperion_analysis import batch_scale_feature
from utils import save_multi_channel_img

def makelist(string):
    delim = ','
    # return [float(_) for _ in string.split(delim)]
    return [_ for _ in string.split(delim)]


def parse_opt():
    parser = argparse.ArgumentParser('Cytof batch process', add_help=False)
    parser.add_argument('--cohort_file', type=str,
                        help='a txt file with information of all file paths in the cohort')
    parser.add_argument('--params_ROI', type=str,
                        help='a txt file with parameters used to process single ROI previously')
    parser.add_argument('--outdir', type=str, help='directory to save outputs')
    parser.add_argument('--save_channel_images', action='store_true',
                        help='an indicator of whether save channel images')
    parser.add_argument('--save_seg_vis', action='store_true',
                        help='an indicator of whether save sample visualization of segmentation')
    parser.add_argument('--show_seg_process', action='store_true',
                        help='an indicator of whether show segmentation process')
    parser.add_argument('--quality_control_thres', type=int, default=50,
                        help='the smallest image size for an image to be kept')
    return parser


def main(args):
    # if args.save_channel_images:
    #     print("saving channel images")
    # else:
    #     print("NOT saving channel images")
    # if args.save_seg_vis:
    #     print("saving segmentation visualization")
    # else:
    #     print("NOT saving segmentation visualization")
    # if args.show_seg_process:
    #     print("showing segmentation process")
    # else:
    #     print("NOT showing segmentation process")
    # parameters used when processing single ROI

    params_ROI   = yaml.load(open(args.params_ROI, "rb"), Loader=yaml.Loader)
    channel_dict = params_ROI["channel_dict"]
    channels_remove = params_ROI["channels_remove"]
    quality_control_thres = params_ROI["quality_control_thres"]

    # name of the batch and saving directory
    cohort_name = os.path.basename(args.cohort_file).split('.csv')[0]
    print(cohort_name)

    outdir = os.path.join(args.outdir, cohort_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    feat_dirs = {}
    feat_dirs['orig'] = os.path.join(outdir, "feature")
    if not os.path.exists(feat_dirs['orig']):
        os.makedirs(feat_dirs['orig'])

    for q in params_ROI["normalize_qs"]:
        dir_qnorm = os.path.join(outdir, f"feature_{q}normed")
        feat_dirs[f"{q}normed"] = dir_qnorm
        if not os.path.exists(dir_qnorm):
            os.makedirs(dir_qnorm)

    dir_img_cytof = os.path.join(outdir, "cytof_images")
    if not os.path.exists(dir_img_cytof):
        os.makedirs(dir_img_cytof)

    if args.save_seg_vis:
        dir_seg_vis = os.path.join(outdir, "segmentation_visualization")
        if not os.path.exists(dir_seg_vis):
            os.makedirs(dir_seg_vis)

    # process batch files
    cohort_files_ = pd.read_csv(args.cohort_file)
    # cohort_files = [os.path.join(cohort_files_.loc[i, "path"], "{}".format(cohort_files_.loc[i, "ROI"])) \
    #                 for i in range(cohort_files_.shape[0])]
    print("Start processing {} files".format(cohort_files_.shape[0]))

    cytof_imgs = {}  # a dictionary contain the full file path of all results
    seen = 0
    dfs_scale_params = {}  # key: quantile q; item: features to be scaled
    df_io = pd.DataFrame(columns=["Slide", "ROI", "path", "output_file"])
    df_bad_rois = pd.DataFrame(columns=["Slide", "ROI", "path", "size (W*H)"])

    # for f_roi in cohort_files:
    for i in range(cohort_files_.shape[0]):
        slide, pth_i, f_roi_ = cohort_files_.loc[i, "Slide"], cohort_files_.loc[i, "path"], cohort_files_.loc[i, "ROI"]
        f_roi = os.path.join(pth_i, f_roi_)
        print("\nNow analyzing {}".format(f_roi))
        roi   = f_roi_.split('.txt')[0]
        print("{}-{}".format(slide, roi))


        ## 1) Read and preprocess data
        # read data: file name -> dataframe
        cytof_img = cytof_read_data_roi(f_roi, slide, roi)

        # quality control section
        cytof_img.quality_control(thres=quality_control_thres)
        if not cytof_img.keep:
            H = max(cytof_img.df['Y'].values) + 1
            W = max(cytof_img.df['X'].values) + 1
        # if (H < args.quality_control_thres) or (W < quality_control_thres):
        #     print("At least one dimension of the image {}-{} is smaller than {}, skipping" \
        #           .format(cytof_img.slide, cytof_img.roi, quality_control_thres))

            df_bad_rois = pd.concat([df_bad_rois,
                                     pd.DataFrame.from_dict([{"Slide": slide,
                                      "ROI": roi,
                                      "path": pth_i,
                                      "size (W*H)": (W,H)}])])
            continue

        if args.save_channel_images:
            dir_roi_channel_img = os.path.join(outdir, "channel_images", f_roi_)
            if not os.path.exists(dir_roi_channel_img):
                os.makedirs(dir_roi_channel_img)

        # markers used when capturing the image
        cytof_img.get_markers()

        # preprocess: fill missing values with 0.
        cytof_img.preprocess()

        # save info
        if seen == 0:
            f_info = open(os.path.join(outdir, 'readme.txt'), 'w')
            f_info.write("Original markers: ")
            f_info.write('\n{}'.format(", ".join(cytof_img.markers)))
            f_info.write("\nOriginal channels: ")
            f_info.write('\n{}'.format(", ".join(cytof_img.channels)))

        ## (optional): save channel images
        if args.save_channel_images:
            cytof_img.get_image()
            cytof_img.save_channel_images(dir_roi_channel_img)

        ## remove special channels if defined
        if len(channels_remove) > 0:
            cytof_img.remove_special_channels(channels_remove)
            cytof_img.get_image()

        ## 2) nuclei & membrane channels and visualization
        cytof_img.define_special_channels(channel_dict)
        assert len(cytof_img.channels) == cytof_img.image.shape[-1]
        # #### Dataframe -> raw image
        # cytof_img.get_image()

        ## (optional): save channel images
        if args.save_channel_images:
            cytof_img.get_image()
            vis_channels = [k for (k, itm) in params_ROI["channel_dict"].items() if len(itm)>0]
            cytof_img.save_channel_images(dir_roi_channel_img, channels=vis_channels)

        ## 3) Nuclei and cell segmentation
        nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=params_ROI["use_membrane"],
                                                 radius=params_ROI["cell_radius"],
                                                 show_process=args.show_seg_process)
        if args.save_seg_vis:
            marked_image_nuclei = cytof_img.visualize_seg(segtype="nuclei", show=False)
            save_multi_channel_img(skimage.img_as_ubyte(marked_image_nuclei[0:100, 0:100, :]),
                                    os.path.join(dir_seg_vis, "{}_{}_nuclei_seg.png".format(slide, roi)))

            marked_image_cell = cytof_img.visualize_seg(segtype="cell", show=False)
            save_multi_channel_img(skimage.img_as_ubyte(marked_image_cell[0:100, 0:100, :]),
                                    os.path.join(dir_seg_vis, "{}_{}_cell_seg.png".format(slide, roi)))

        ## 4) Feature extraction
        cytof_img.extract_features(f_roi)

        # save the original extracted feature
        cytof_img.df_feature.to_csv(os.path.join(feat_dirs['orig'], "{}_{}_feature_summary.csv".format(slide, roi)),
                                    index=False)

        ### 4.1) Log transform and quantile normalization
        cytof_img.feature_quantile_normalization(qs=params_ROI["normalize_qs"], savedir=feat_dirs['orig'])

        # calculate scaling parameters
        ## features to be scaled
        if seen == 0:
            s_features = [col for key, features in cytof_img.features.items() \
                      for f in features \
                      for col in cytof_img.df_feature.columns if col.startswith(f)]

            f_info.write("\nChannels removed: ")
            f_info.write("\n{}".format(", ".join(channels_remove)))
            f_info.write("\nFinal markers: ")
            f_info.write("\n{}".format(', '.join(cytof_img.markers)))
            f_info.write("\nFinal channels: ")
            f_info.write("\n{}".format(', '.join(cytof_img.channels)))
            f_info.close()
        ## loop over quantiles
        for q, quantile in cytof_img.dict_quantiles.items():
            n_attr = f"df_feature_{q}normed"
            df_normed = getattr(cytof_img, n_attr)
            # save the normalized features to csv
            df_normed.to_csv(os.path.join(feat_dirs[f"{q}normed"],
                                          "{}_{}_feature_summary.csv".format(slide, roi)),
                             index=False)
            if seen == 0:
                dfs_scale_params[q] = df_normed[s_features]
                dict_quantiles = cytof_img.dict_quantiles
            else:
                # dfs_scale_params[q] = dfs_scale_params[q].append(df_normed[s_features], ignore_index=True)
                dfs_scale_params[q] = pd.concat([dfs_scale_params[q], df_normed[s_features]])

        seen += 1

        # save the class instance
        out_file = os.path.join(dir_img_cytof, "{}_{}.pkl".format(slide, roi))
        cytof_img.save_cytof(out_file)
        cytof_imgs[roi] = out_file
        # df_io = df_io.append({"Slide": slide,
        #                       "ROI": roi,
        #                       "path": pth_i,
        #                       "output_file": out_file}, ignore_index=True)
        df_io = pd.concat([df_io,
                           pd.DataFrame.from_dict([{"Slide": slide,
                            "ROI": roi,
                            "path": pth_i,
                            "output_file": os.path.abspath(out_file) # use absolute path
                            }])
                           ])


    for q in dict_quantiles.keys():
        df_scale_params = dfs_scale_params[q].mean().to_frame(name="mean").transpose()
        # df_scale_params = df_scale_params.append(dfs_scale_params[q].std().to_frame(name="std").transpose(),
        #                                          ignore_index=True)
        df_scale_params = pd.concat([df_scale_params, dfs_scale_params[q].std().to_frame(name="std").transpose()])
        df_scale_params.to_csv(os.path.join(outdir, f"{q}normed_scale_params.csv"), index=False)


    # df_io = pd.DataFrame.from_dict(cytof_imgs, orient="index", columns=['output_file'])
    # df_io.reset_index(inplace=True)
    # df_io.rename(columns={'index': 'input_file'}, inplace=True)
    df_io.to_csv(os.path.join(outdir, "input_output.csv"), index=False)
    if len(df_bad_rois) > 0:
        df_bad_rois.to_csv(os.path.join(outdir, "skipped_rois.csv"), index=False)

    # scale feature
    batch_scale_feature(outdir, normqs=params_ROI["normalize_qs"], df_io=df_io)
    # return cytof_imgs, feat_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cytof batch process', parents=[parse_opt()])
    args  = parser.parse_args()
    main(args)
