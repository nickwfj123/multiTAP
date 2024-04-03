import os
import re
import glob
import pickle as pkl

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import warnings
from tqdm import tqdm
import skimage

import phenograph
import umap
import seaborn as sns
from scipy.stats import spearmanr

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

import hyperion_preprocess as pre
import hyperion_segmentation as seg
from utils import load_CytofImage

# from cytof import hyperion_preprocess as pre
# from cytof import hyperion_segmentation as seg
# from cytof.utils import load_CytofImage





def _longest_substring(str1, str2):
    ans = ""
    len1, len2 = len(str1), len(str2)
    for i in range(len1):
        for j in range(len2):
            match = ""
            _len = 0
            while ((i+_len < len1) and (j+_len < len2) and str1[i+_len] == str2[j+_len]):
                match += str1[i+_len]
                _len += 1
                if len(match) > len(ans):
                    ans = match
    return ans

def extract_feature(channels, raw_image, nuclei_seg, cell_seg, filename, show_head=False):
    """ Extract nuclei and cell level feature from cytof image based on nuclei segmentation and cell segmentation
        results
    Inputs:
        channels   = channels to extract feature from
        raw_image  = raw cytof image
        nuclei_seg = nuclei segmentation result
        cell_seg   = cell segmentation result
        filename   = filename of current cytof image
    Returns:
        feature_summary_df = a dataframe containing summary of extracted features
        morphology         = names of morphology features extracted

    :param channels: list
    :param raw_image: numpy.ndarray
    :param nuclei_seg: numpy.ndarray
    :param cell_seg: numpy.ndarray
    :param filename: string
    :param morpholoty: list
    :return feature_summary_df: pandas.core.frame.DataFrame
    """
    assert (len(channels) == raw_image.shape[-1])

    # morphology features to be extracted
    morphology = ["area", "convex_area", "eccentricity", "extent",
                "filled_area", "major_axis_length", "minor_axis_length",
                "orientation", "perimeter", "solidity", "pa_ratio"]

    ## morphology features
    nuclei_morphology = [_ + '_nuclei' for _ in morphology]  # morphology - nuclei level
    cell_morphology = [_ + '_cell' for _ in morphology]  # morphology - cell level

    ## single cell features
    # nuclei level
    sum_exp_nuclei = [_ + '_nuclei_sum' for _ in channels]  # sum expression over nuclei
    ave_exp_nuclei = [_ + '_nuclei_ave' for _ in channels]  # average expression over nuclei

    # cell level
    sum_exp_cell   = [_ + '_cell_sum' for _ in channels]  # sum expression over cell
    ave_exp_cell   = [_ + '_cell_ave' for _ in channels]  # average expression over cell

    # column names of final result dataframe
    column_names       = ["filename", "id", "coordinate_x", "coordinate_y"] + \
                         sum_exp_nuclei + ave_exp_nuclei + nuclei_morphology + \
                         sum_exp_cell + ave_exp_cell + cell_morphology

    # Initiate
    res = dict()
    for column_name in column_names:
        res[column_name] = []

    n_nuclei = np.max(nuclei_seg)
    for nuclei_id in tqdm(range(2, n_nuclei + 1), position=0, leave=True):
        res["filename"].append(filename)
        res["id"].append(nuclei_id)
        regions = skimage.measure.regionprops((nuclei_seg == nuclei_id) * 1)  # , coordinates='xy') (deprecated)
        if len(regions) >= 1:
            this_nucleus = regions[0]
        else:
            continue
        regions = skimage.measure.regionprops((cell_seg == nuclei_id) * 1)  # , coordinates='xy') (deprecated)
        if len(regions) >= 1:
            this_cell = regions[0]
        else:
            continue
        centroid_y, centroid_x = this_nucleus.centroid  # y: rows; x: columns
        res['coordinate_x'].append(centroid_x)
        res['coordinate_y'].append(centroid_y)

        # morphology
        for i, feature in enumerate(morphology[:-1]):
            res[nuclei_morphology[i]].append(getattr(this_nucleus, feature))
            res[cell_morphology[i]].append(getattr(this_cell, feature))
        res[nuclei_morphology[-1]].append(1.0 * this_nucleus.perimeter ** 2 / this_nucleus.filled_area)
        res[cell_morphology[-1]].append(1.0 * this_cell.perimeter ** 2 / this_cell.filled_area)

        # markers
        for i, marker in enumerate(channels):
            ch = i
            res[sum_exp_nuclei[i]].append(np.sum(raw_image[nuclei_seg == nuclei_id, ch]))
            res[ave_exp_nuclei[i]].append(np.average(raw_image[nuclei_seg == nuclei_id, ch]))
            res[sum_exp_cell[i]].append(np.sum(raw_image[cell_seg == nuclei_id, ch]))
            res[ave_exp_cell[i]].append(np.average(raw_image[cell_seg == nuclei_id, ch]))

    feature_summary_df = pd.DataFrame(res)
    if show_head:
        print(feature_summary_df.head())
    return feature_summary_df


###############################################################################
# def check_feature_distribution(feature_summary_df, features):
#     """ Visualize feature distribution for each feature
#     Inputs:
#         feature_summary_df = dataframe of extracted feature summary
#         features           = features to check distribution
#     Returns:
#         None

#     :param feature_summary_df: pandas.core.frame.DataFrame
#     :param features: list
#     """

#     for feature in features:
#         print(feature)
#         fig, ax = plt.subplots(1, 1, figsize=(3, 2))
#         ax.hist(np.log2(feature_summary_df[feature] + 0.0001), 100)
#         ax.set_xlim(-15, 15)
#         plt.show()



def feature_quantile_normalization(feature_summary_df, features, qs=[75,99]):
    """ Calculate the q-quantiles of selected features given quantile q values. Then perform q-quantile normalization
     on these features using calculated quantile values. The feature_summary_df will be updated in-place with new
     columns "feature_qnormed" generated and added. Meanwhile, visualize distribution of log2 features before and after
     q-normalization
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to be normalized
        qs                 = quantile q values (default=[75,99])
    Returns:
        quantiles          = quantile values for each q
    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param qs: list
    :return quantiles: dict
    """
    expressions = []
    expressions_normed = dict((key, []) for key in qs)
    quantiles   = {}
    colors = cm.rainbow(np.linspace(0, 1, len(qs)))
    for feat in features:
        quantiles[feat] = {}
        expressions.extend(feature_summary_df[feat])

        plt.hist(np.log2(np.array(expressions) + 0.0001), 100, density=True)
        for q, c in zip(qs, colors):
            quantile_val = np.quantile(expressions, q/100)
            quantiles[feat][q] = quantile_val
            plt.axvline(np.log2(quantile_val), label=f"{q}th percentile", c=c)
            print(f"{q}th percentile: {quantile_val}")

            # log-quantile normalization
            normed = np.log2(feature_summary_df.loc[:, feat] / quantile_val + 0.0001)
            feature_summary_df.loc[:, f"{feat}_{q}normed"] = normed
            expressions_normed[q].extend(normed)
        plt.xlim(-15, 15)
        plt.xlabel("log2(expression of all markers)")
        plt.legend()
        plt.show()

    # visualize before & after quantile normalization
    '''N = len(qs)+1 # (len(qs)+1) // 2 + (len(qs)+1) %2'''
    log_expressions = tuple([np.log2(np.array(expressions) + 0.0001)] + [expressions_normed[q] for q in qs])
    labels = ["before normalization"] + [f"after {q} normalization" for q in qs]
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.hist(log_expressions, 100, density=True, label=labels)
    ax.set_xlabel("log2(expressions for all markers)")
    plt.legend()
    plt.show()
    return quantiles


def feature_scaling(feature_summary_df, features, inplace=False):
    """Perform in-place mean-std scaling on selected features. Normally, do not scale nuclei sum feature
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to perform scaling on
        inplace            = an indicator of whether perform the scaling in-place (Default=False)
    Returns:

    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param inplace: bool
    """

    scaled_feature_summary_df = feature_summary_df if inplace else feature_summary_df.copy()

    for feat in features:
        if feat not in feature_summary_df.columns:
            print(f"Warning: {feat} not available!")
            continue
        scaled_feature_summary_df[feat] = \
            (scaled_feature_summary_df[feat] - np.average(scaled_feature_summary_df[feat])) \
            / np.std(scaled_feature_summary_df[feat])
    if not inplace:
        return scaled_feature_summary_df






def generate_summary(feature_summary_df, features, thresholds):
    """Generate (cell level) summary table for each feature in features: feature name, total number (of cells),
        calculated GMM threshold for this feature, number of individuals (cells) with greater than threshold values,
        ratio of individuals (cells) with greater than threshold values
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = a list of features to generate summary table
        thresholds         = (calculated GMM-based) thresholds for each feature
    Outputs:
        df_info    = summary table for each feature

    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param thresholds: dict
    :return df_info: pandas.core.frame.DataFrame
    """

    df_info = pd.DataFrame(columns=['feature', 'total number', 'threshold', 'positive counts', 'positive ratio'])
    
    for feature in features:
        # calculate threshold
        thres = thresholds[feature]
        X = feature_summary_df[feature].values
        n = sum(X > thres)
        N = len(X)

        df_new_row = pd.DataFrame({'feature': feature,'total number':N, 'threshold':thres,
                                  'positive counts':n, 'positive ratio': n/N}, index=[0])
        df_info = pd.concat([df_info, df_new_row])
    return df_info


# def visualize_thresholding_outcome(feat,
#                                    feature_summary_df,
#                                    raw_image,
#                                    channel_names,
#                                    thres,
#                                    nuclei_seg,
#                                    cell_seg,
#                                    vis_quantile_q=0.9, savepath=None):
#     """ Visualize calculated threshold for a feature by mapping back to nuclei and cell segmentation outputs - showing
#         greater than threshold pixels in red color, others with blue color.
#         Meanwhile, visualize the original image with red color indicating the channel correspond to the feature.
#     Inputs:
#         feat               = name of the feature to visualize
#         feature_summary_df = dataframe of extracted feature summary
#         raw_image          = raw cytof image
#         channel_names       = a list of marker names, which is consistent with each channel in the raw_image
#         thres              = threshold value for feature "feat"
#         nuclei_seg         = nuclei segmentation output
#         cell_seg           = cell segmentation output
#     Outputs:
#         stain_nuclei       = nuclei segmentation output stained with threshold information
#         stain_cell         = cell segmentation output stained with threshold information
#     :param feat: string
#     :param feature_summary_df: pandas.core.frame.DataFrame
#     :param raw_image: numpy.ndarray
#     :param channel_names: list
#     :param thres: float
#     :param nuclei_seg: numpy.ndarray
#     :param cell_seg: numpy.ndarray
#     :return stain_nuclei: numpy.ndarray
#     :return stain_cell: numpy.ndarray
#     """
#     col_name = channel_names[np.argmax([len(_longest_substring(feat, x)) for x in channel_names])]
#     col_id   = channel_names.index(col_name)
#     df_temp = pd.DataFrame(columns=[f"{feat}_overthres"], data=np.zeros(len(feature_summary_df), dtype=np.int32))
#     df_temp.loc[feature_summary_df[feat] > thres, f"{feat}_overthres"] = 1
#     feature_summary_df = pd.concat([feature_summary_df, df_temp], axis=1)
#     # feature_summary_df.loc[:, f"{feat}_overthres"] = 0
#     # feature_summary_df.loc[feature_summary_df[feat] > thres, f"{feat}_overthres"] = 1
#
#     '''rgba_color = [plt.cm.get_cmap('tab20').colors[_ % 20] for _ in feature_summary_df.loc[:, f"{feat}_overthres"]]'''
#     color_ids  = []
#
#     # stained Nuclei image
#     stain_nuclei = np.zeros((nuclei_seg.shape[0], nuclei_seg.shape[1], 3)) + 1
#     for i in range(2, np.max(nuclei_seg) + 1):
#         color_id = feature_summary_df[f"{feat}_overthres"][feature_summary_df['id'] == i].values[0] * 2
#         if color_id not in color_ids:
#             color_ids.append(color_id)
#         stain_nuclei[nuclei_seg == i] = plt.cm.get_cmap('tab20').colors[color_id][:3]
#
#     # stained Cell image
#     stain_cell = np.zeros((cell_seg.shape[0], cell_seg.shape[1], 3)) + 1
#     for i in range(2, np.max(cell_seg) + 1):
#         color_id = feature_summary_df[f"{feat}_overthres"][feature_summary_df['id'] == i].values[0] * 2
#         stain_cell[cell_seg == i] = plt.cm.get_cmap('tab20').colors[color_id][:3]
#
#     fig, axs = plt.subplots(1,3,figsize=(16, 8))
#     if col_id != 0:
#         channel_ids = (col_id, 0)
#     else:
#         channel_ids = (col_id, -1)
#     '''print(channel_ids)'''
#     quantiles = [np.quantile(raw_image[..., _], vis_quantile_q) for _ in channel_ids]
#     vis_img, _ = pre.cytof_merge_channels(raw_image, channel_names=channel_names,
#                                           channel_ids=channel_ids, quantiles=quantiles)
#     marker = feat.split("(")[0]
#     print(f"Nuclei and cell with high {marker} expression shown in orange, low in blue.")
#
#     axs[0].imshow(vis_img)
#     axs[1].imshow(stain_nuclei)
#     axs[2].imshow(stain_cell)
#     axs[0].set_title("pseudo-colored original image")
#     axs[1].set_title(f"{marker} expression shown in nuclei")
#     axs[2].set_title(f"{marker} expression shown in cell")
#     if savepath is not None:
#         plt.savefig(savepath)
#     plt.show()
#     return stain_nuclei, stain_cell, vis_img


########################################################################################################################
############################################### batch functions ########################################################
########################################################################################################################
def batch_extract_feature(files, markers, nuclei_markers, membrane_markers=None, show_vis=False):
    """Extract features for cytof images from a list of files. Normally this list contains ROIs of the same slide
    Inputs:
        files            = a list of files to be processed
        markers          = a list of marker names used when generating the image
        nuclei_markers   = a list of markers define the nuclei channel (used for nuclei segmentation)
        membrane_markers = a list of markers define the membrane channel (used for cell segmentation) (Default=None)
        show_vis         = an indicator of showing visualization during process
    Outputs:
        file_features    = a dictionary contains extracted features for each file

    :param files: list
    :param markers: list
    :param nuclei_markers: list
    :param membrane_markers: list
    :param show_vis: bool
    :return file_features: dict
    """
    file_features = {}
    for f in tqdm(files):
        # read data
        df = pre.cytof_read_data(f)
        # preprocess
        df_ = pre.cytof_preprocess(df)
        column_names = markers[:]
        df_output = pre.define_special_channel(df_, 'nuclei', markers=nuclei_markers)
        column_names.insert(0, 'nuclei')
        if membrane_markers is not None:
            df_output = pre.define_special_channel(df_output, 'membrane', markers=membrane_markers)
            column_names.append('membrane')
        raw_image = pre.cytof_txt2img(df_output, marker_names=column_names)

        if show_vis:
            merged_im, _ = pre.cytof_merge_channels(raw_image, channel_ids=[0, -1], quantiles=None, visualize=False)
            plt.imshow(merged_im[0:200, 200:400, ...])
            plt.title('Selected region of raw cytof image')
            plt.show()

        # nuclei and cell segmentation
        nuclei_img = raw_image[..., column_names.index('nuclei')]
        nuclei_seg, color_dict = seg.cytof_nuclei_segmentation(nuclei_img, show_process=False)
        if membrane_markers is not None:
            membrane_img = raw_image[..., column_names.index('membrane')]
            cell_seg, _ = seg.cytof_cell_segmentation(nuclei_seg, membrane_channel=membrane_img, show_process=False)
        else:
            cell_seg, _ = seg.cytof_cell_segmentation(nuclei_seg, show_process=False)
        if show_vis:
            marked_image_nuclei = seg.visualize_segmentation(raw_image, nuclei_seg, channel_ids=(0, -1), show=False)
            marked_image_cell = seg.visualize_segmentation(raw_image, cell_seg, channel_ids=(-1, 0), show=False)
            fig, axs = plt.subplots(1,2,figsize=(10,6))
            axs[0].imshow(marked_image_nuclei[0:200, 200:400, :]), axs[0].set_title('nuclei segmentation')
            axs[1].imshow(marked_image_cell[0:200, 200:400, :]), axs[1].set_title('cell segmentation')
            plt.show()

        # feature extraction
        feat_names = markers[:]
        feat_names.insert(0, 'nuclei')
        df_feat_sum = extract_feature(feat_names, raw_image, nuclei_seg, cell_seg, filename=f)
        file_features[f] = df_feat_sum
    return file_features



def batch_norm_scale(file_features, column_names, qs=[75,99]):
    """Perform feature log transform, quantile normalization and scaling in a batch
    Inputs:
        file_features = A dictionary of dataframes containing extracted features. key - file name, item - feature table
        column_names  = A list of markers. Should be consistent with column names in dataframe of features
        qs            = quantile q values (Default=[75,99])
    Outputs:
        file_features_out = log transformed, quantile normalized and scaled features for each file in the batch
        quantiles         = a dictionary of quantile values for each file in the batch

    :param file_features: dict
    :param column_names: list
    :param qs: list
    :return file_features_out: dict
    :return quantiles: dict
    """
    file_features_out = copy.deepcopy(file_features) # maintain a copy of original file_features

    # marker features
    cell_markers_sum   = [_ + '_cell_sum' for _ in column_names]
    cell_markers_ave   = [_ + '_cell_ave' for _ in column_names]
    nuclei_markers_sum = [_ + '_nuclei_sum' for _ in column_names]
    nuclei_markers_ave = [_ + '_nuclei_ave' for _ in column_names]

    # morphology features
    morphology = ["area", "convex_area", "eccentricity", "extent",
                  "filled_area", "major_axis_length", "minor_axis_length",
                  "orientation", "perimeter", "solidity", "pa_ratio"]
    nuclei_morphology = [_ + '_nuclei' for _ in morphology]  # morphology - nuclei level
    cell_morphology   = [_ + '_cell' for _ in morphology]  # morphology - cell level

    # features to be normalized
    features_to_norm = [x for x in nuclei_markers_sum + nuclei_markers_ave + cell_markers_sum + cell_markers_ave \
                        if not x.startswith('nuclei')]

    # features to be scaled
    scale_features = []
    for feature_name in nuclei_morphology + cell_morphology + nuclei_markers_sum + nuclei_markers_ave + \
                        cell_markers_sum + cell_markers_ave:
        '''if feature_name not in nuclei_morphology + cell_morphology and not feature_name.startswith('nuclei'):
            scale_features += [feature_name, f"{feature_name}_75normed", f"{feature_name}_99normed"]
        else:
            scale_features += [feature_name]'''
        temp = [feature_name]
        if feature_name not in nuclei_morphology + cell_morphology and not feature_name.startswith('nuclei'):
            for q in qs:
                temp += [f"{feature_name}_{q}normed"]
        scale_features += temp

    quantiles = {}
    for f, df in file_features_out.items():
        print(f)
        quantiles[f] = feature_quantile_normalization(df, features=features_to_norm, qs=qs)
        feature_scaling(df, features=scale_features, inplace=True)
    return file_features_out, quantiles


def batch_scale_feature(outdir, normqs, df_io=None, files_scale=None):
    """
    Inputs:
        outdir      = output saving directory, which contains the scale file generated previously,
                     the input_output_csv file with the list of available cytof_img class instances in the batch,
                     as well as previously saved cytof_img class instances in .pkl files
        normqs      = a list of q values of percentile normalization
        files_scale = full file name of the scaling information

    Outputs: None
        Scaled feature are saved as .csv files in subfolder "feature_qnormed_scaled" in outdir
        A new attribute will be added to cytof_img class instance, and the update class instance is saved in outdir
    """
    if df_io is None:
        df_io = pd.read_csv(os.path.join(outdir, "input_output.csv"))

    for _i, normq in enumerate(normqs):
        n_attr = f"df_feature_{normq}normed"
        n_attr_scaled = f"{n_attr}_scaled"
        file_scale = files_scale[_i] if files_scale is not None else os.path.join(outdir, "{}normed_scale_params.csv".format(normq))
        # saving directory of scaled normed feature
        dirq = os.path.join(outdir, f"feature_{normq}normed_scaled")
        if not os.path.exists(dirq):
            os.makedirs(dirq)

        # load scaling parameters
        df_scale = pd.read_csv(file_scale, index_col=False)
        m = df_scale[df_scale.columns].iloc[0] # mean
        s = df_scale[df_scale.columns].iloc[1] # std.dev

        dfs = {}
        cytofs = {}
        # save scaled feature
        for f_cytof in df_io['output_file']:
    #     for roi, f_cytof in zip(df_io['ROI'], df_io['output_file']):
            cytof_img = pkl.load(open(f_cytof, "rb"))
            assert hasattr(cytof_img, n_attr), f"attribute {n_attr} not exist"
            df_feat = copy.deepcopy(getattr(cytof_img, n_attr))

            assert len([x for x in df_scale.columns if x not in df_feat.columns]) == 0

            # scale
            df_feat[df_scale.columns] = (df_feat[df_scale.columns] - m) / s

            # save scaled feature to csv
            df_feat.to_csv(os.path.join(dirq, os.path.basename(f_cytof).replace('.pkl', '.csv')), index=False)

            # add attribute "df_feature_scaled"
            setattr(cytof_img, n_attr_scaled, df_feat)

            # save updated cytof_img class instance
            pkl.dump(cytof_img, open(f_cytof, "wb"))


def batch_generate_summary(outdir, feature_type="normed", normq=75, scaled=True, vis_thres=False):
    """
    Inputs:
        outdir       = output saving directory, which contains the scale file generated previously, as well as previously saved
                     cytof_img class instances in .pkl files
        feature_type = type of feature to be used, available choices: "original", "normed", "scaled"
        normq        = q value of quantile normalization
        scaled       = a flag indicating whether or not use the scaled version of features (Default=False)
        vis_thres    = a flag indicating whether or not visualize the process of calculating thresholds (Default=False)
    Outputs: None
        Two .csv files, one for cell sum and the other for cell average features, are saved for each ROI, containing the
        threshold and cell count information of each feature, in the subfolder "marker_summary" under outdir
    """
    assert feature_type in ["original", "normed", "scaled"], 'accepted feature types are "original", "normed", "scaled"'
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"{normq}normed"
    else:
        feat_name = f"{normq}normed_scaled"

    n_attr = f"df_feature_{feat_name}"

    dir_sum = os.path.join(outdir, "marker_summary", feat_name)
    print(dir_sum)
    if not os.path.exists(dir_sum):
        os.makedirs(dir_sum)

    seen = 0
    dfs = {}
    cytofs = {}
    df_io  = pd.read_csv(os.path.join(outdir, "input_output.csv"))
    for f in df_io['output_file'].tolist():
        f_roi = os.path.basename(f).split(".pkl")[0]
        cytof_img = pkl.load(open(f, "rb"))

        ##### updated #####
        df_feat = getattr(cytof_img, n_attr)
        dfs[f]  = getattr(cytof_img, n_attr)
        cytofs[f] = cytof_img
        ##### end updated #####

        if seen == 0:
            feat_cell_sum = cytof_img.features['cell_sum']
            feat_cell_ave = cytof_img.features['cell_ave']
        seen += 1

    ##### updated #####  
    all_df     = pd.concat(dfs.values(), ignore_index=True)
    print("Getting thresholds for marker sum")
    thres_sum = _get_thresholds(all_df, feat_cell_sum, visualize=vis_thres)
    print("Getting thresholds for marker average")
    thres_ave = _get_thresholds(all_df, feat_cell_ave, visualize=vis_thres)
    for f, cytof_img in cytofs.items():
        f_roi = os.path.basename(f).split(".pkl")[0]
        df_info_cell_sum_f = generate_summary(dfs[f], features=feat_cell_sum, thresholds=thres_sum)
        df_info_cell_ave_f = generate_summary(dfs[f], features=feat_cell_ave, thresholds=thres_ave)
        setattr(cytof_img, f"cell_count_{feat_name}_sum", df_info_cell_sum_f)
        setattr(cytof_img, f"cell_count_{feat_name}_ave", df_info_cell_ave_f)
        df_info_cell_sum_f.to_csv(os.path.join(dir_sum, f"{f_roi}_cell_count_sum.csv"), index=False)
        df_info_cell_ave_f.to_csv(os.path.join(dir_sum, f"{f_roi}_cell_count_ave.csv"), index=False)
        pkl.dump(cytof_img, open(f, "wb"))        
    return dir_sum



def _gather_roi_expressions(df_io, normqs=[75]):
    """Only cell level sum"""
    expressions = {}
    expressions_normed = {}
    for roi in df_io["ROI"].unique():
        expressions[roi] = []
        f_cytof_im = df_io.loc[df_io["ROI"] == roi, "output_file"].values[0]
        cytof_im = load_CytofImage(f_cytof_im)
        for feature_name in cytof_im.features['cell_sum']:
            expressions[roi].extend(cytof_im.df_feature[feature_name])
        expressions_normed[roi] = dict((q, {}) for q in normqs)
        for q in expressions_normed[roi].keys():
            expressions_normed[roi][q] = []
            normed_feat = getattr(cytof_im, "df_feature_{}normed".format(q))
            for feature_name in cytof_im.features['cell_sum']:
                expressions_normed[roi][q].extend(normed_feat[feature_name])
    return expressions, expressions_normed


def visualize_normalization(df_slide_roi, normqs=[75], level="slide"):
    expressions_, expressions_normed_ = _gather_roi_expressions(df_slide_roi, normqs=normqs)
    if level == "slide":
        prefix = "Slide"
        expressions, expressions_normed = {}, {}
        for slide in df_slide_roi["Slide"].unique():
            f_rois = df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"].values
            rois = [x.replace('.txt', '') for x in f_rois]
            expressions[slide] = []
            expressions_normed[slide] = dict((q, []) for q in normqs)
            for roi in rois:
                expressions[slide].extend(expressions_[roi])

                for q in expressions_normed[slide].keys():
                    expressions_normed[slide][q].extend(expressions_normed_[roi][q])

    else:
        expressions, expressions_normed = expressions_, expressions_normed_
        prefix = "ROI"
    num_q = len(normqs)
    for key, key_exp in expressions.items():  # create a new plot for each slide (or ROI)
        print("Showing {} {}".format(prefix, key))
        fig, ax = plt.subplots(1, num_q + 1, figsize=(4 * (num_q + 1), 4))
        ax[0].hist((np.log2(np.array(key_exp) + 0.0001),), 100, density=True)
        ax[0].set_title("Before normalization")
        ax[0].set_xlabel("log2(cellular expression of all markers)")
        for i, q in enumerate(normqs):
            ax[i + 1].hist((np.array(expressions_normed[key][q]) + 0.0001,), 100, density=True)
            ax[i + 1].set_title("After {}-th percentile normalization".format(q))
            ax[i + 1].set_xlabel("log2(cellular expression of all markers)")
        plt.show()
    return expressions, expressions_normed


###########################################################
############# marker level analysis functions #############
###########################################################

############# marker co-expression analysis #############
def _gather_roi_co_exp(df_slide_roi, outdir, feat_name, accumul_type):
    """roi level co-expression analysis"""
    n_attr = f"df_feature_{feat_name}"
    expected_percentages = {}
    edge_percentages     = {}
    num_cells            = {}

    for seen_roi, f_roi in enumerate(df_slide_roi["ROI"].unique()):
        roi = f_roi.replace(".txt", "")
        slide = df_slide_roi.loc[df_slide_roi["ROI"] == f_roi, "Slide"].values[0]
        f_cytof_im = "{}_{}.pkl".format(slide, roi)
        if not f_cytof_im in os.listdir(os.path.join(outdir, "cytof_images")):
            print("{} not found, skip".format(f_cytof_im))
            continue
        cytof_im   = load_CytofImage(os.path.join(outdir, "cytof_images", f_cytof_im))
        df_feat = getattr(cytof_im, n_attr)

        if seen_roi == 0:
            # all gene (marker) columns
            marker_col_all = [x for x in df_feat.columns if "cell_{}".format(accumul_type) in x]
            marker_all = [x.split('(')[0] for x in marker_col_all]
            n_marker = len(marker_col_all)  
        n_cell   = len(df_feat)
        # corresponding marker positive info file
        df_info_cell = getattr(cytof_im,"cell_count_{}_{}".format(feat_name,accumul_type))
        pos_nums   = df_info_cell["positive counts"].values
        pos_ratios = df_info_cell["positive ratio"].values
        thresholds = df_info_cell["threshold"].values

        # create new expected_percentage matrix for each ROI
        expected_percentage = np.zeros((n_marker, n_marker))

        # expected_percentage
        # an N by N matrix, where N represent for the number of total gene (marker)
        # each ij-th element represents for the percentage that both the i-th and the j-th gene is "positive"
        # based on the threshold defined previously

        for ii in range(n_marker):
            for jj in range(n_marker):
                expected_percentage[ii, jj] = pos_nums[ii] * pos_nums[jj]
        expected_percentages[roi] = expected_percentage      
        # edge_percentage
        # an N by N matrix, where N represent for the number of gene (marker)
        # each ij-th element represents for the percentage of cells that show positive in both i-th and j-th gene
        edge_nums = np.zeros_like(expected_percentage)
        for ii in range(n_marker):
            _x = df_feat[marker_col_all[ii]].values > thresholds[ii] # _x = df_feat[marker_col_all[ii]].values > thresholds[marker_idx[ii]]
            for jj in range(n_marker):
                _y = df_feat[marker_col_all[jj]].values > thresholds[jj] # _y = df_feat[marker_col_all[jj]].values > thresholds[marker_idx[jj]]
                edge_nums[ii, jj] = np.sum(np.all([_x, _y], axis=0)) # / n_cell
        edge_percentages[roi] = edge_nums
        num_cells[roi]        = n_cell
    return expected_percentages, edge_percentages, num_cells, marker_all, marker_col_all


def co_expression_analysis(df_slide_roi, outdir, feature_type, accumul_type, co_exp_markers="all", normq=75,
                           level="slide", clustergrid=None):
    """
    """
    assert level in ["slide", "roi"], "Only slide or roi levels are accepted!"
    assert feature_type in ["original", "normed", "scaled"]
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"{normq}normed"
    else:
        feat_name = f"{normq}normed_scaled"

    print(feat_name)
    dir_cytof_img = os.path.join(outdir, "cytof_images")

    expected_percentages, edge_percentages, num_cells, marker_all, marker_col_all = \
        _gather_roi_co_exp(df_slide_roi, outdir, feat_name, accumul_type)

    if co_exp_markers != "all":
        # assert (isinstance(co_exp_markers, list) and all([x in cytof_img.markers for x in co_exp_markers]))
        assert (isinstance(co_exp_markers, list) and all([x in marker_all for x in co_exp_markers]))
        marker_idx = np.array([marker_all.index(x) for x in co_exp_markers])
        marker_all = [marker_all[x] for x in marker_idx]
        marker_col_all = [marker_col_all[x] for x in marker_idx]
    else:
        marker_idx = np.arange(len(marker_all))

    if level == "slide":
        # expected_percentages, edge_percentages = {}, {}
        for slide in df_slide_roi["Slide"].unique():  ## for each slide
            for seen_roi, f_roi in enumerate(df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]):  ## for each ROI
                roi = f_roi.replace(".txt", "")
                if roi not in expected_percentages:
                    continue
                if seen_roi == 0:
                    expected_percentages[slide] = expected_percentages[roi]
                    edge_percentages[slide] = edge_percentages[roi]
                    num_cells[slide] = num_cells[roi]
                else:
                    expected_percentages[slide] += expected_percentages[roi]
                    edge_percentages[slide] += edge_percentages[roi]
                    num_cells[slide] += num_cells[roi]
                expected_percentages.pop(roi)
                edge_percentages.pop(roi)
                num_cells.pop(roi)

    co_exps = {}
    for key, expected_percentage in expected_percentages.items():
        expected_percentage = expected_percentage / num_cells[key] ** 2
        edge_percentage = edge_percentages[key] / num_cells[key]

        # Normalize
        edge_percentage_norm = np.log10(edge_percentage / expected_percentage + 0.1)

        # Fix Nan
        edge_percentage_norm[np.isnan(edge_percentage_norm)] = np.log10(1 + 0.1)

        co_exps[key] = edge_percentage_norm

    # plot
    for f_key, edge_percentage_norm in co_exps.items():
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(edge_percentage_norm[marker_idx, :][:, marker_idx], center=np.log10(1 + 0.1),
                         # ax = sns.heatmap(edge_percentage_norm, center=np.log10(1 + 0.1),
                         cmap='RdBu_r', vmin=-1, vmax=3,
                         xticklabels=marker_all, yticklabels=marker_all)
        ax.set_aspect('equal')
        plt.title(f_key)
        plt.show()

        if clustergrid is None:
            plt.figure()
            clustergrid = sns.clustermap(edge_percentage_norm[marker_idx, :][:, marker_idx],
                                         # clustergrid = sns.clustermap(edge_percentage_norm,
                                         center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=3,
                                         xticklabels=marker_all, yticklabels=marker_all, figsize=(6, 6))
            plt.title(f_key)
            plt.show()

        # else:
        plt.figure()
        sns.clustermap(edge_percentage_norm[marker_idx, :][:, marker_idx] \
                       # sns.clustermap(edge_percentage_norm \
                       [clustergrid.dendrogram_row.reordered_ind, :][:, clustergrid.dendrogram_row.reordered_ind],
                       center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=3,
                       xticklabels=np.array(marker_all)[clustergrid.dendrogram_row.reordered_ind],
                       yticklabels=np.array(marker_all)[clustergrid.dendrogram_row.reordered_ind],
                       figsize=(6, 6), row_cluster=False, col_cluster=False)
        plt.title(f_key)
        plt.show()
    return co_exps, marker_idx, clustergrid

############# marker correlation #############
from scipy.stats import spearmanr

def _gather_roi_corr(df_slide_roi, outdir, feat_name, accumul_type):
    """roi level correlation analysis"""
    
    n_attr = f"df_feature_{feat_name}"
    feats = {}

    for seen_roi, f_roi in enumerate(df_slide_roi["ROI"].unique()):## for each ROI
        roi = f_roi.replace(".txt", "")
        slide = df_slide_roi.loc[df_slide_roi["ROI"] == f_roi, "Slide"].values[0]
        f_cytof_im = "{}_{}.pkl".format(slide, roi)
        if not f_cytof_im in os.listdir(os.path.join(outdir, "cytof_images")):
            print("{} not found, skip".format(f_cytof_im))
            continue
        cytof_im   = load_CytofImage(os.path.join(outdir, "cytof_images", f_cytof_im))
        df_feat    = getattr(cytof_im, n_attr)
        feats[roi] = df_feat
        
        if seen_roi == 0:
            # all gene (marker) columns
            marker_col_all = [x for x in df_feat.columns if "cell_{}".format(accumul_type) in x]
            marker_all = [x.split('(')[0] for x in marker_col_all]
    return feats, marker_all, marker_col_all


def correlation_analysis(df_slide_roi, outdir, feature_type, accumul_type, corr_markers="all", normq=75, level="slide",
                         clustergrid=None):
    """
    """
    assert level in ["slide", "roi"], "Only slide or roi levels are accepted!"
    assert feature_type in ["original", "normed", "scaled"]
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"{normq}normed"
    else:
        feat_name = f"{normq}normed_scaled"

    print(feat_name)
    dir_cytof_img = os.path.join(outdir, "cytof_images")

    feats, marker_all, marker_col_all = _gather_roi_corr(df_slide_roi, outdir, feat_name, accumul_type)
    n_marker = len(marker_all)

    corrs = {}
    # n_marker = len(marker_all)
    if level == "slide":
        for slide in df_slide_roi["Slide"].unique():  ## for each slide
            for seen_roi, f_roi in enumerate(df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]):  ## for each ROI
                roi = f_roi.replace(".txt", "")
                if roi not in feats:
                    continue
                if seen_roi == 0:
                    feats[slide] = feats[roi]
                else:
                    #                     feats[slide] = feats[slide].append(feats[roi], ignore_index=True)
                    feats[slide] = pd.concat([feats[slide], feats[roi]])
                feats.pop(roi)

    for key, feat in feats.items():
        correlation = np.zeros((n_marker, n_marker))
        for i, feature_i in enumerate(marker_col_all):
            for j, feature_j in enumerate(marker_col_all):
                correlation[i, j] = spearmanr(feat[feature_i].values, feat[feature_j].values).correlation
        corrs[key] = correlation

    if corr_markers != "all":
        assert (isinstance(corr_markers, list) and all([x in marker_all for x in corr_markers]))
        marker_idx = np.array([marker_all.index(x) for x in corr_markers])
        marker_all = [marker_all[x] for x in marker_idx]
        marker_col_all = [marker_col_all[x] for x in marker_idx]
    else:
        marker_idx = np.arange(len(marker_all))

    # plot
    for f_key, corr in corrs.items():
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(corr[marker_idx, :][:, marker_idx], center=np.log10(1 + 0.1),
                         cmap='RdBu_r', vmin=-1, vmax=1,
                         xticklabels=corr_markers, yticklabels=corr_markers)
        ax.set_aspect('equal')
        plt.title(f_key)
        plt.show()

        if clustergrid is None:
            plt.figure()
            clustergrid = sns.clustermap(corr[marker_idx, :][:, marker_idx],
                                         center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
                                         xticklabels=corr_markers, yticklabels=corr_markers, figsize=(6, 6))
            plt.title(f_key)
            plt.show()

        plt.figure()
        sns.clustermap(corr[marker_idx, :][:, marker_idx] \
                           [clustergrid.dendrogram_row.reordered_ind, :][:, clustergrid.dendrogram_row.reordered_ind],
                       center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
                       xticklabels=np.array(corr_markers)[clustergrid.dendrogram_row.reordered_ind],
                       yticklabels=np.array(corr_markers)[clustergrid.dendrogram_row.reordered_ind],
                       figsize=(6, 6), row_cluster=False, col_cluster=False)
        plt.title(f_key)
        plt.show()
    return corrs, marker_idx, clustergrid

############# marker interaction #############

from sklearn.neighbors import DistanceMetric
from tqdm import tqdm

def _gather_roi_interact(df_slide_roi, outdir, feat_name, accumul_type, interact_markers="all", thres_dist=50):
    dist = DistanceMetric.get_metric('euclidean')
    n_attr = f"df_feature_{feat_name}"
    edge_percentages = {}
    num_edges = {}
    for seen_roi, f_roi in enumerate(df_slide_roi["ROI"].unique()):  ## for each ROI
        roi = f_roi.replace(".txt", "")
        slide = df_slide_roi.loc[df_slide_roi["ROI"] == f_roi, "Slide"].values[0]
        f_cytof_im = "{}_{}.pkl".format(slide, roi)
        if not f_cytof_im in os.listdir(os.path.join(outdir, "cytof_images")):
            print("{} not found, skip".format(f_cytof_im))
            continue
        cytof_im = load_CytofImage(os.path.join(outdir, "cytof_images", f_cytof_im))
        df_feat  = getattr(cytof_im, n_attr)
        n_cell   = len(df_feat)
        dist_matrix = dist.pairwise(df_feat.loc[:, ['coordinate_x', 'coordinate_y']].values)

        if seen_roi==0:
            # all gene (marker) columns
            marker_col_all = [x for x in df_feat.columns if "cell_{}".format(accumul_type) in x]
            marker_all = [x.split('(')[0] for x in marker_col_all]    
            n_marker = len(marker_col_all) 
            
        # corresponding marker positive info file
        df_info_cell = getattr(cytof_im,"cell_count_{}_{}".format(feat_name,accumul_type))
        thresholds   = df_info_cell["threshold"].values#[marker_idx]
            
        n_edges = 0
        # expected_percentage = np.zeros((n_marker, n_marker))
        # edge_percentage = np.zeros_like(expected_percentage)
        edge_nums = np.zeros((n_marker, n_marker))    
        # interaction
        cluster_sub = []
        for i_cell in range(n_cell):
            _temp = set()   
            for k in range(n_marker):
                if df_feat[marker_col_all[k]].values[i_cell] > thresholds[k]:
                    _temp = _temp | {k}
            cluster_sub.append(_temp)

        for i in tqdm(range(n_cell)):
            for j in range(n_cell):
                 if dist_matrix[i, j] > 0 and dist_matrix[i, j] < thres_dist:
                        n_edges += 1
                        for m in cluster_sub[i]:
                            for n in cluster_sub[j]:
                                edge_nums[m, n] += 1
                                
        edge_percentages[roi] = edge_nums#/n_edges 
        num_edges[roi] = n_edges
    return edge_percentages, num_edges, marker_all, marker_col_all


def interaction_analysis(df_slide_roi,
                         outdir,
                         feature_type,
                         accumul_type,
                         interact_markers="all",
                         normq=75,
                         level="slide",
                         thres_dist=50,
                         clustergrid=None):
    """
    """
    assert level in ["slide", "roi"], "Only slide or roi levels are accepted!"
    assert feature_type in ["original", "normed", "scaled"]
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"{normq}normed"
    else:
        feat_name = f"{normq}normed_scaled"

    print(feat_name)
    dir_cytof_img = os.path.join(outdir, "cytof_images")

    expected_percentages, _, num_cells, marker_all_, marker_col_all_ = \
        _gather_roi_co_exp(df_slide_roi, outdir, feat_name, accumul_type)
    edge_percentages, num_edges, marker_all, marker_col_all = \
        _gather_roi_interact(df_slide_roi, outdir, feat_name, accumul_type, interact_markers="all",
                             thres_dist=thres_dist)

    if level == "slide":
        for slide in df_slide_roi["Slide"].unique():  ## for each slide
            for seen_roi, f_roi in enumerate(df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]):  ## for each ROI
                roi = f_roi.replace(".txt", "")
                if roi not in expected_percentages:
                    continue
                if seen_roi == 0:
                    expected_percentages[slide] = expected_percentages[roi]
                    edge_percentages[slide] = edge_percentages[roi]
                    num_edges[slide] = num_edges[roi]
                    num_cells[slide] = num_cells[roi]
                else:
                    expected_percentages[slide] += expected_percentages[roi]
                    edge_percentages[slide] += edge_percentages[roi]
                    num_edges[slide] += num_edges[roi]
                    num_cells[slide] += num_cells[roi]
                expected_percentages.pop(roi)
                edge_percentages.pop(roi)
                num_edges.pop(roi)
                num_cells.pop(roi)

    if interact_markers != "all":
        assert (isinstance(interact_markers, list) and all([x in marker_all for x in interact_markers]))
        marker_idx = np.array([marker_all.index(x) for x in interact_markers])
        marker_all = [marker_all[x] for x in marker_idx]
        marker_col_all = [marker_col_all[x] for x in marker_idx]
    else:
        marker_idx = np.arange(len(marker_all))

    interacts = {}
    for key, edge_percentage in edge_percentages.items():
        expected_percentage = expected_percentages[key] / num_cells[key] ** 2
        edge_percentage = edge_percentage / num_edges[key]

        # Normalize
        edge_percentage_norm = np.log10(edge_percentage / expected_percentage + 0.1)

        # Fix Nan
        edge_percentage_norm[np.isnan(edge_percentage_norm)] = np.log10(1 + 0.1)
        interacts[key] = edge_percentage_norm

    # plot
    for f_key, interact_ in interacts.items():
        interact = interact_[marker_idx, :][:, marker_idx]
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(interact, center=np.log10(1 + 0.1),
                         cmap='RdBu_r', vmin=-1, vmax=1,
                         xticklabels=interact_markers, yticklabels=interact_markers)
        ax.set_aspect('equal')
        plt.title(f_key)
        plt.show()

        if clustergrid is None:
            plt.figure()
            clustergrid = sns.clustermap(interact, center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
                                         xticklabels=interact_markers, yticklabels=interact_markers, figsize=(6, 6))
            plt.title(f_key)
            plt.show()

        plt.figure()
        sns.clustermap(
            interact[clustergrid.dendrogram_row.reordered_ind, :][:, clustergrid.dendrogram_row.reordered_ind],
            center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
            xticklabels=np.array(interact_markers)[clustergrid.dendrogram_row.reordered_ind],
            yticklabels=np.array(interact_markers)[clustergrid.dendrogram_row.reordered_ind],
            figsize=(6, 6), row_cluster=False, col_cluster=False)
        plt.title(f_key)
        plt.show()
    return interacts, clustergrid

###########################################################
######## Pheno-Graph clustering analysis functions ########
###########################################################

def clustering_phenograph(cohort_file, outdir, normq=75, feat_comb="all", k=None, save_vis=False, pheno_markers="all"):
    """Perform Pheno-graph clustering for the cohort
        Inputs:
            cohort_file   = a .csv file include the whole cohort
            outdir        = output saving directory, previously saved cytof_img class instances in .pkl files
            normq         = q value for quantile normalization
            feat_comb     = desired feature combination to be used for phenograph clustering, acceptable choices: "all",
                        "cell_sum", "cell_ave", "cell_sum_only", "cell_ave_only" (Default="all")
            k             = number of initial neighbors to run Pheno-graph (Default=None)
                        If k is not provided, k is set to N / 100, where N is the total number of single cells
            save_vis      = a flag indicating whether to save the visualization output (Default=False)
            pheno_markers = a list of markers used in phenograph clustering (must be a subset of cytof_img.markers)
    Outputs:
        df_all     = a dataframe of features for all cells in the cohort, with the clustering output saved in the column
        'phenotype_total{n_community}', where n_community stands for the total number of communities defined by the cohort
            Also, each individual cytof_img class instances will be updated with 2 new attributes:
            1)"num phenotypes ({feat_comb}_{normq}normed_{k})"
            2)"phenotypes ({feat_comb}_{normq}normed_{k})"
        feat_names  = feature names (columns) used to generate PhenoGraph output
        k           = the initial number of k used to run PhenoGraph
        pheno_name  = the column name of the added column indicating phenograph cluster 
        vis_savedir = the directory to save the visualization output
        markers     = the list of markers used (minimal, for visualization purposes)
    """
    
    vis_savedir = ""
    feat_groups = {
        "all": ["cell_sum", "cell_ave", "cell_morphology"],
        "cell_sum": ["cell_sum", "cell_morphology"],
        "cell_ave": ["cell_ave", "cell_morphology"],
        "cell_sum_only": ["cell_sum"],
        "cell_ave_only": ["cell_ave"]
    }
    assert feat_comb in feat_groups.keys(), f"{feat_comb} not supported!"

    feat_name = f"_{normq}normed_scaled"
    n_attr    = f"df_feature{feat_name}"

    dfs = {}
    cytof_ims = {}

    df_io = pd.read_csv(os.path.join(outdir, "input_output.csv"))
    df_slide_roi = pd.read_csv(cohort_file)

    # load all scaled feature in the cohort
    for i in df_io.index:
        f_out = df_io.loc[i, "output_file"]
        f_roi = f_out.split('/')[-1].split('.pkl')[0]
        if not os.path.isfile(f_out):
            print("{} not found, skip".format(f_out))
            continue

        cytof_img = load_CytofImage(f_out)
        if i == 0:
            dict_feat = cytof_img.features
            markers   = cytof_img.markers
        cytof_ims[f_roi] = cytof_img
        dfs[f_roi] = getattr(cytof_img, n_attr)

    feat_names = []
    for y in feat_groups[feat_comb]:
        if "morphology" in y:
            feat_names += dict_feat[y]
        else:
            if pheno_markers == "all":
                feat_names += dict_feat[y]
                pheno_markers = markers
            else:
                assert isinstance(pheno_markers, list)
                ids = [markers.index(x) for x in pheno_markers]
                feat_names += [dict_feat[y][x] for x in ids]
    # concatenate feature dataframes of all rois in the cohort
    df_all = pd.concat([_ for key, _ in dfs.items()])
    
    # set number of nearest neighbors k and run PhenoGraph for phenotype clustering
    k = k if k else int(df_all.shape[0] / 100)  # 100
    communities, graph, Q = phenograph.cluster(df_all[feat_names], k=k, n_jobs=-1)  # run PhenoGraph
    n_community = len(np.unique(communities))

    # Visualize
    ## project to 2D
    umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(df_all[feat_names])

    # plot together
    print("Visualization in 2d - cohort")
    plt.figure(figsize=(4, 4))
    plt.title("cohort")
    sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=communities, palette='tab20',
                    #                 legend=legend,
                    hue_order=np.arange(n_community))
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    if save_vis:
        vis_savedir = os.path.join(outdir, "phenograph_{}_{}normed_{}".format(feat_comb, normq, k))
        if not os.path.exists(vis_savedir):
            os.makedirs(vis_savedir)
        plt.savefig(os.path.join(vis_savedir, "cluster_scatter.png"))
    plt.show()

    # attach clustering output to df_all
    pheno_name = f'phenotype_total{n_community}'
    df_all[pheno_name] = communities
    df_all['{}_projx'.format(pheno_name)] = proj_2d[:,0]
    df_all['{}_projy'.format(pheno_name)] = proj_2d[:,1]
    return df_all, feat_names, k, pheno_name, vis_savedir, markers
    
    
def _gather_roi_pheno(df_slide_roi, df_all):
    """Split whole df into df for each ROI"""
    pheno_roi = {}      
    
    for i in df_slide_roi.index:
        path_i = df_slide_roi.loc[i, "path"]
        roi_i  = df_slide_roi.loc[i, "ROI"]
        f_in  = os.path.join(path_i, roi_i)
        cond  = df_all["filename"] == f_in
        pheno_roi[roi_i.replace(".txt", "")] = df_all.loc[cond, :] 
    return pheno_roi  
    
    
def _vis_cell_phenotypes(df_feat, communities, n_community, markers, list_features, accumul_type="sum", savedir=None, savename=""):
    """ Visualize cell phenotypes for a given dataframe of feature
    Args:
        df_feat: a dataframe of features
        communities: a list of communities (can be a subset of the cohort communities, but should be consistent with df_feat)
        n_community: number of communities in the cohort (n_community >= number of unique values in communities)
        markers: a list of markers used in CyTOF image (to be present in the heatmap visualization)
        list_features: a list of feature names (consistent with columns in df_feat)
        accumul_type: feature aggregation type, choose from "sum" and "ave" (default="sum")
        savedir: results saving directory. If not None, visualization plots will be saved in the desired directory (default=None)
    Returns:
        cell_cluster: a (N, M) matrix, where N = # of clustered communities, and M = # of markers

        cell_cluster_norm: the normalized form of cell_cluster (normalized by subtracting the median value)
    """
    assert accumul_type in ["sum", "ave"], "Wrong accumulation type! Choose from 'sum' and 'ave'!"
    cell_cluster = np.zeros((n_community, len(markers)))
    for cluster in range(len(np.unique(communities))):
        df_sub = df_feat[communities == cluster]
        if df_sub.shape[0] == 0:
            continue
        
        for i, feat in enumerate(list_features): # for each feature in the list of features
            cell_cluster[cluster, i] = np.average(df_sub[feat])
    cell_cluster_norm = cell_cluster - np.median(cell_cluster, axis=0) 
    sns.heatmap(cell_cluster_norm, # cell_cluster - np.median(cell_cluster, axis=0),#
                cmap='magma',
                xticklabels=markers,
                yticklabels=np.arange(len(np.unique(communities)))
               )
    plt.xlabel("Markers - {}".format(accumul_type))
    plt.ylabel("Phenograph clusters")
    plt.title("normalized expression - cell {}".format(accumul_type))
    savename += "_cell_{}.png".format(accumul_type)
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, savename))
    plt.show()
    return cell_cluster, cell_cluster_norm
    
def vis_phenograph(df_slide_roi, df_all, pheno_name, markers, used_feat, level="cohort", accumul_type="sum", 
                   to_save=False, savepath="./", vis_scatter=False):
    """
    Args:
        df_slide_roi = a dataframe with slide-roi correspondence information included
        df_all       = dataframe with feature and clustering results included
        pheno_name   = name (key) of the phenograph output
        markers      = a (minimal) list of markers used in Pheno-Graph (to visualize)
        list_feat    = a list of features used (should be consistent with columns available in df_all)
        level        = level to visualize, choose from "cohort", "slide", or "roi" (default="cohort")
        accumul_type = type of feature accumulation used (default="sum")
        to_save      = a flag indicating whether or not save output (default=False)
        savepath     = visualization saving directory (default="./")
    """
    if to_save:
        if not os.path.exists(savepath):
            os.makedirs
            
    # features used for accumul_type 
    ids       = [i for (i,x) in enumerate(used_feat) if re.search(".{}".format(accumul_type), x)]
    list_feat = [used_feat[i] for i in ids]

    '''# features used for cell ave
    accumul_type             = "ave"
    ids                      = [i for (i,x) in enumerate(used_feats[key]) if re.search(".{}".format(accumul_type), x)]
    list_feats[accumul_type] = [used_feats[key][i] for i in ids]

    list_feat_morph = [x for x in used_feats[key] if x not in list_feats["sum"]+list_feats["ave"]]'''

    if accumul_type == "sum":
        suffix = "_cell_sum"
    elif accumul_type == "ave":
        suffix = "_cell_ave"

    assert level in ["cohort", "slide", "roi"], "Only 'cohort', 'slide' or 'roi' levels are accepted!"
    '''df_io = pd.read_csv(os.path.join(outdir, "input_output.csv"))'''
    
    n_community = len(df_all[pheno_name].unique())
    if level == "cohort":
        phenos = {level: df_all}
    else:
        phenos = _gather_roi_pheno(df_slide_roi, df_all)
        if level == "slide":
            for slide in df_io["Slide"].unique(): # for each slide
                for seen_roi, roi_i in enumerate(df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]):  ## for each ROI
                    
                    f_roi = roi_i.replace(".txt", "")
                    if seen_roi == 0:
                        phenos[slide] = phenos[f_roi]
                    else:
                        phenos[slide] = pd.concat([phenos[slide], phenos[f_roi]])
                    phenos.pop(f_roi)
    
    
    savename = ""
    for key, df_pheno in phenos.items():
        if to_save:
            savepath_ = os.path.join(savepath, level)
            savename = key
        communities = df_pheno[pheno_name]
        
        _vis_cell_phenotypes(df_pheno, communities, n_community, markers, list_feat, accumul_type, 
                             savedir=savepath_, savename=savename)
        
        # visualize scatter (2-d projection)
        if vis_scatter: 
            proj_2d = df_pheno[['{}_projx'.format(pheno_name), '{}_projy'.format(pheno_name)]].to_numpy()
#             print("Visualization in 2d - cohort")
            plt.figure(figsize=(4, 4))
            plt.title("cohort")
            sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=communities, palette='tab20',
                            #                 legend=legend,
                            hue_order=np.arange(n_community))
            plt.axis('tight')
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
            if to_save:
                plt.savefig(os.path.join(savepath_, "scatter_{}.png".format(savename)))
            plt.show()
    return phenos
    

import sklearn.neighbors
from sklearn.neighbors import kneighbors_graph as skgraph
from sklearn.metrics import DistanceMetric# from sklearn.neighbors import DistanceMetric
from scipy import sparse as sp
import networkx as nx

def _gather_roi_distances(df_slide_roi, outdir, name_pheno, thres_dist=50):
    dist = DistanceMetric.get_metric('euclidean')
    dist_matrices = {}
    for i, f_roi in enumerate(df_slide_roi['ROI'].unique()):
        roi = f_roi.replace('.txt', '')
        slide = df_slide_roi.loc[df_slide_roi["ROI"] == f_roi, "Slide"].values[0]
        f_cytof_im = "{}_{}.pkl".format(slide, roi)
        if not f_cytof_im in os.listdir(os.path.join(outdir, "cytof_images")):
            print("{} not found, skip".format(f_cytof_im))
            continue
        cytof_im   = load_CytofImage(os.path.join(outdir, "cytof_images", f_cytof_im))
        df_sub = cytof_im.df_feature
        dist_matrices[roi] = {}
        dist_matrices[roi]['dist'] = dist.pairwise(df_sub.loc[:, ['coordinate_x', 'coordinate_y']].values)
    
        phenograph = getattr(cytof_im, 'phenograph')[name_pheno]
        cluster = phenograph['clusters'].values

        if i == 0:
            n_cluster = phenograph['num_community']

        # expected percentage
        expected_percentage = np.zeros((n_cluster, n_cluster))
        for _i in range(n_cluster):
            for _j in range(n_cluster):
                expected_percentage[_i, _j] = sum(cluster == _i) * sum(cluster == _j) #/ len(df_sub)**2
        dist_matrices[roi]['expected_percentage'] = expected_percentage
        dist_matrices[roi]['num_cell'] = len(df_sub)

        # edge num
        edge_nums = np.zeros_like(expected_percentage)
        dist_matrix = dist_matrices[roi]['dist']
        n_cells = dist_matrix.shape[0]
        for _i in range(n_cells):
            for _j in range(n_cells):
                 if dist_matrix[_i, _j] > 0 and dist_matrix[_i, _j] < thres_dist:
                        edge_nums[cluster[_i], cluster[_j]] += 1
        # edge_percentages = edge_nums/np.sum(edge_nums)
        dist_matrices[roi]['edge_nums'] = edge_nums
    return dist_matrices


def _gather_roi_kneighbor_graphs(df_slide_roi, outdir, name_pheno, k=8):
    graphs = {}
    for i, f_roi in enumerate(df_slide_roi['ROI'].unique()):
        roi = f_roi.replace('.txt', '')
        f_cytof_im = "{}.pkl".format(roi)
        if not f_cytof_im in os.listdir(os.path.join(outdir, "cytof_images")):
            print("{} not found, skip".format(f_cytof_im))
            continue
        cytof_im   = load_CytofImage(os.path.join(outdir, "cytof_images", f_cytof_im))
        df_sub = cytof_im.df_feature
        graph = skgraph(np.array(df_sub.loc[:, ['coordinate_x', 'coordinate_y']]), n_neighbors=k, mode='distance')
        graph.toarray()
        I, J, V = sp.find(graph)

        graphs[roi] = {}
        graphs[roi]['I'] = I  # Start (center)
        graphs[roi]['J'] = J  # End
        graphs[roi]['V'] = V
        graphs[roi]['graph'] = graph

        phenograph = getattr(cytof_im, 'phenograph')[name_pheno]
        cluster    = phenograph['clusters'].values

        if i == 0:
            n_cluster = phenograph['num_community']

        # Edge type summary
        edge_nums = np.zeros((n_cluster, n_cluster))
        for _i, _j in zip(I, J):
            edge_nums[cluster[_i], cluster[_j]] += 1
        graphs[roi]['edge_nums'] = edge_nums
        '''edge_percentages = edge_nums/np.sum(edge_nums)'''

        expected_percentage = np.zeros((n_cluster, n_cluster))
        for _i in range(n_cluster):
            for _j in range(n_cluster):
                expected_percentage[_i, _j] = sum(cluster == _i) * sum(cluster == _j) #/ len(df_sub)**2
        graphs[roi]['expected_percentage'] = expected_percentage
        graphs[roi]['num_cell'] = len(df_sub)
    return graphs


def interaction_analysis(df_slide_roi, outdir, name_pheno, method="distance", k=8, thres_dist=50, level="slide", clustergrid=None):
    assert method in ["distance", "graph"], "Method can be either 'distance' or 'graph'!"
    
    if method == "distance":
        info = _gather_roi_distances(df_slide_roi, outdir, name_pheno, thres_dist)
    else:
        info = _gather_roi_kneighbor_graphs(df_slide_roi, outdir, name_pheno, k)
        
    interacts = {}
    if level == "slide":
        for slide in df_slide_roi["Slide"].unique():
            for seen_roi, f_roi in enumerate(df_slide_roi.loc[df_slide_roi["Slide"] == slide, "ROI"]):
                roi = f_roi.replace(".txt", "")
                if seen_roi == 0:
                    info[slide] = {}
                    info[slide]['edge_nums']           = info[roi]['edge_nums']
                    info[slide]['expected_percentage'] = info[roi]['expected_percentage']
                    info[slide]['num_cell']            = info[roi]['num_cell']

                else:
                    info[slide]['edge_nums']           += info[roi]['edge_nums']
                    info[slide]['expected_percentage'] += info[roi]['expected_percentage']
                    info[slide]['num_cell']            += info[roi]['num_cell']
                info.pop(roi)

    for key, item in info.items():
        edge_percentage     = item['edge_nums'] / np.sum(item['edge_nums'])
        expected_percentage = item['expected_percentage'] / item['num_cell'] ** 2

        # Normalize
        interact_norm       = np.log10(edge_percentage/expected_percentage + 0.1)

        # Fix Nan
        interact_norm[np.isnan(interact_norm)] = np.log10(1 + 0.1)
        interacts[key]      = interact_norm

    # plot
    for f_key, interact in interacts.items():
        plt.figure(figsize=(6, 6))
        ax = sns.heatmap(interact, center=np.log10(1 + 0.1),
                         cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_aspect('equal')
        plt.title(f_key)
        plt.show()

        if clustergrid is None:
            plt.figure()
            clustergrid = sns.clustermap(interact, center=np.log10(1 + 0.1), 
                                         cmap='RdBu_r', vmin=-1, vmax=1, 
                                         xticklabels=np.arange(interact.shape[0]), 
                                         yticklabels=np.arange(interact.shape[0]), 
                                         figsize=(6, 6))
            plt.title(f_key)
            plt.show()

        plt.figure()
        sns.clustermap(interact[clustergrid.dendrogram_row.reordered_ind, :]\
                       [:, clustergrid.dendrogram_row.reordered_ind],
                       center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=1,
                       xticklabels=clustergrid.dendrogram_row.reordered_ind, 
                       yticklabels=clustergrid.dendrogram_row.reordered_ind,
                       figsize=(6, 6), row_cluster=False, col_cluster=False)
        plt.title(f_key)
        plt.show()

    return interacts, clustergrid