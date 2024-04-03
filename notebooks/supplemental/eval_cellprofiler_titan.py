import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import skimage
import yaml
from typing import Union, Optional, Type, Tuple, List, Dict
import sys
import pandas as pd
import nrrd
import json

# Project Root
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# used for searching packages and functions

ROOT_DIR = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/image_cytof'
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'image_cytof'))
from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import save_multi_channel_img, check_feature_distribution

IMC_FOLDER = '/archive/DPDS/Xiao_lab/shared/shidan/hyperion/The Single-Cell Pathology Landscape of Breast Cancer/OMEandSingleCellMasks/OMEnMasks/ome/ome'
IMC_MASKS_FOLDER = '/archive/DPDS/Xiao_lab/shared/shidan/hyperion/The Single-Cell Pathology Landscape of Breast Cancer/OMEandSingleCellMasks/OMEnMasks/Basel_Zuri_masks/Basel_Zuri_masks'
IMC_MARKER_FILE = '/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/github/image_cytof_test_data/external_data/The Single-Cell Pathology Landscape of Breast Cancer/markers_labels.txt'
BATCH_NUM = 350
CP_MASK_PATH = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/image_cytof/notebooks/developer/roi_masks/cellprofiler/'
TITAN_MASK_PATH = '/project/DPDS/Xiao_lab/shared/deep_learning_SW_RR/cytof/image_cytof/notebooks/developer/roi_masks/titan/nrrd_keep'
CELL_RADIUS = 5
# define the nuclei and membrane markers
channel_dict = {
        'nuclei': ['DNA1-Ir191', 'DNA2-Ir193'],
        'membrane': ['Vimentin-Sm149', 'c-erbB-2 - Her2-Eu151', 'pan Cytokeratin-Keratin Epithelial-Lu175', 
                     'CD44-Gd160','Fibronectin-Nd142'], 
    }


def get_iou_matrix(y_true, y_pred, pred_bg=None, true_bg=None):
    '''
    Calculates mIOU of cell segmentation predictions.

    @param y_true: ground truth of how cells are segmented
           y_pred: model prediciton of cell segmentations
           pred_bg: integer, optional. the background label in prediction
           true_bg: integer. the background label in ground truth

    @return exclude_bg_iou_matrix: the performance matrix for cell segmentations. col: ground truth, row: predicted labels
    '''
    
    assert true_bg is not None, 'background not specified. Identify the background label in prediction and/or in ground truth.'

    y_true, y_pred = y_true.flatten(), y_pred.flatten()

    # returns a sorted list of unique values
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    uni_true, uni_pred = np.unique(y_true), np.unique(y_pred)

    # find where membrane labels are in the sorted arrays
    true_bg_index = np.where(uni_true==true_bg)[0][0]
    pred_bg_index = np.where(uni_pred==pred_bg)[0][0] if pred_bg is not None else None

    nb_true, nb_pred = len(uni_true), len(uni_pred)
    print(f"{nb_true-1} cells found in ground truth; {nb_pred-1} cells found in prediction.")
    
    # Calculate A_and_B, A_or_B and iou_score
    # for y_true and y_pred, count the number of overlaps
    A_and_B = np.histogram2d(y_true, y_pred, bins=(nb_true, nb_pred))[0]

    # np.histogram returns hist and the edges. [0] to get only the histogram count
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html

    # -1 in reshape specifies unknown dimention, and 1 here forces num_rows=1
    # ref: https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape

    # bins set to nb_true(num of unique label) simply creates a count for each label
    # bins defines the number of equal-width bins (in increasing order)
    A_or_B = (np.histogram(y_true, bins = nb_true)[0].reshape((-1, 1))
              + np.histogram(y_pred, bins = nb_pred)[0]
              - A_and_B) + 1e-8
    
    # iou_matrix guarantees ascending cell labels in matrix
    iou_matrix = (A_and_B / A_or_B).T # transforms so that each column is ground truth, row is predicted labels
    exclude_bg_iou_matrix = np.delete(iou_matrix, obj=true_bg_index, axis=1) # remove background in ground truth

    if pred_bg_index is not None:
        exclude_bg_iou_matrix = np.delete(exclude_bg_iou_matrix, obj=pred_bg_index, axis=0) # remove background in pred
   
    # if there are no predictions available, create a (1, nb_true) size matrix to be compatible with benchmarking
    if len(exclude_bg_iou_matrix) == 0:
        exclude_bg_iou_matrix = np.zeros((1, nb_true))
    return exclude_bg_iou_matrix


def generate_iou(iou_matrix, threshold):
    '''
    Given the IOU matrix, calcualte coverage and mIOU at threshold
    @param iou_matrix: the performance matrix for cell segmentations
           threshold: the threshold each IOU need to pass in order to be considered a match

    @return iou_list: the list of IOUs that passed threshold
            len_ground_truth: number of true cells labels 
    '''
    # order columns of iou_matrix, before matching with pred
    # convert to df for indexing
    df_temp = pd.DataFrame(iou_matrix)
    max_index = df_temp.max().sort_values(ascending=False).index
    iou_matrix = df_temp.reindex(columns=max_index).values

    # find maximum IOU in each ground truth that predictions matched with the largest IOU, and remove GT from next match
    # ref: https://stackoverflow.com/questions/69922997/finding-the-index-of-max-value-of-columns-in-numpy-array-but-removing-the-previo
    max_iou = np.array([])

    len_ground_truth = iou_matrix.shape[1]
    for j in range(len_ground_truth):

        # find the maximum index for each column
        idx = np.argmax(iou_matrix[:, j])
        
        # add IOU to list
        max_iou = np.append(max_iou, iou_matrix[idx, j])
        
        # replace with negative inf for all values in that row
        # next argmax will ignore this row 
        iou_matrix[idx, :] = -np.inf
        
#     print('max_iou:', max_iou)
    matched_iou = max_iou[max_iou >= threshold] # filter out any IOU less than threshold
#     print('matched_iou:', matched_iou)
    
    return matched_iou, len_ground_truth

# load the roi -> imc mapping file
roi_name_dict = json.load(open('roi_num_imc_name_map.json'))

# initialize an array of IoU dictionaries
iou_matrix_schema = np.array([])

# used to set the markers after reading the file
labels_markers = yaml.load(open(IMC_MARKER_FILE, "rb"), Loader=yaml.Loader)

# process all IMC files with CP masks
for roi_num in range(BATCH_NUM):
  print('PROCESSING ROI', roi_num)
  # find the corresponding IMC file
  imc_file = roi_name_dict['rois/roi'+str(roi_num)]

  # read the IMC file into CytofImageTIFF class
  full_file_path = os.path.join(IMC_FOLDER, imc_file)
  cytof_img, _ = cytof_read_data_roi(full_file_path, slide='slide', roi=roi_num)
  
  # set markers for further analysis
  cytof_img.set_markers(**labels_markers)

  # define special channels
  channels_rm = cytof_img.define_special_channels(channel_dict, rm_key='nuclei')

  # # remove unwanted channels
  cytof_img.remove_special_channels(channels_rm)


  nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=True, 
                                         radius=CELL_RADIUS,
                                         show_process=False)
  
  # store predicted cell segmentation
  cell_seg_pred = cytof_img.cell_seg - 1 # to match with gt label

  # load the CP masks result
  cell_seg_cellprofiler = skimage.io.imread(os.path.join(CP_MASK_PATH, f'roi{roi_num}_cell_masks.tiff'), plugin="tifffile")
  
  # load the TITAN masks result
  cell_seg_titan, _ = nrrd.read(os.path.join(TITAN_MASK_PATH, f'roi{roi_num} Cell Mask.nrrd'))
  cell_seg_titan = np.transpose(cell_seg_titan, axes=(1,0,2)) # TITAN processing always transposes the original images

  # load ground truth
  cell_seg_gt = skimage.io.imread(os.path.join(IMC_MASKS_FOLDER, imc_file.split('.tiff')[0]+"_maks.tiff"), plugin='tifffile')
  
  # sometimes the ground truth shape does not match with the original image shape. If that's the case, transpose the image
  if cell_seg_gt.shape != cytof_img.image.shape[0:2]:
    cell_seg_gt = cell_seg_gt.T
  
  print('dimensions check', cytof_img.image.shape, cell_seg_pred.shape, cell_seg_cellprofiler.shape, cell_seg_gt.shape, cell_seg_titan.shape)


  # store the results into a dictionary
  res_dict = {'Ours':cell_seg_pred, 'CellProfiler':cell_seg_cellprofiler, 'TITAN': cell_seg_titan}
  res_iou_dict = dict()

  # compute the matched IoU between prediction and gt for each method
  for res_key in res_dict.keys():
    y_pred = res_dict[res_key]
    iou_matrix = get_iou_matrix(cell_seg_gt, y_pred, pred_bg=0, true_bg=0)
    res_iou_dict[res_key] = iou_matrix

  # each element in iou_matrix_schema contains a dict of each ROI's performance. the dict contains each method's performance
  iou_matrix_schema = np.append(iou_matrix_schema, res_iou_dict)

assert len(iou_matrix_schema) == BATCH_NUM

# going through iou schema, average matched IOU and calculate coverage as needed
perform_dict = dict()

iou_thresh = np.arange(0.01,1, step=0.01)

# print(iou_matrix_schema)

for thresh in iou_thresh:
    print('Evaluating threshold:', thresh)
    iou_matched_dict = dict()
    num_matched_dict = dict()
    num_gt_total_dict = dict()
    num_prediction_dict = dict()
    
    # going through each result dictionary, get matched iou for that threshold
    # type(iou_matrix) is dict()
    for res_dict in iou_matrix_schema: # res_dict are iou metrics
        for res_key in res_dict.keys():
            
            # getting iou matric for particular watershed method
            iou_matrix = res_dict[res_key]
        
            # calculate interested metrics
            # empty prediction. algorithm cannot segment any cells
            matched_ind_iou, len_gt = generate_iou(iou_matrix, thresh)
            
            # find total prediction
            num_pred = iou_matrix.shape[0]
            
            # first time seeing res_key, initialize list to record matched IOU, if any, to the list
            # only occurs in the first res_dict object
            if res_key not in iou_matched_dict.keys(): 
                iou_matched_dict[res_key] = []
                num_matched_dict[res_key] = 0
                num_gt_total_dict[res_key] = 0
                num_prediction_dict[res_key] = 0
                
                
            # add all elements in matched_ind_iou to iou_matched
            iou_matched_dict[res_key].extend(matched_ind_iou)

            # extracted number of matched predictions
            num_matched_dict[res_key] += len(matched_ind_iou)

            # add up number of ground truths in each img
            num_gt_total_dict[res_key] += len_gt
            
            # add total number of predictions
            num_prediction_dict[res_key] += num_pred
            
    # summarize at the dataset level
    # should only have keys like mm_basin, no_mm_basin, color_decov, etc
    for res_key in iou_matched_dict.keys():
        miou = np.mean(iou_matched_dict[res_key]) # miou for the entire dataset at thresh 0.01 for method mm_basin
        try:
            cov = num_matched_dict[res_key] / num_gt_total_dict[res_key]
            precision = num_matched_dict[res_key] / num_prediction_dict[res_key]
            f1_score = 2*precision*cov / (precision+cov)
        except ZeroDivisionError:
            print(f'Zero division during calculation at threshold {thresh}, iteration skipped.')
            continue # if zero division, do not append values
        
        # first time recording performance dict (i.e. at the first threshold)
        if res_key not in perform_dict.keys():
            perform_dict[res_key] = ([miou], [cov], [precision], [f1_score])
            
        # results can be mapped to the tuple
        else:
            perform_dict[res_key][0].append(miou)
            perform_dict[res_key][1].append(cov)
            perform_dict[res_key][2].append(precision)
            perform_dict[res_key][3].append(f1_score)

# save the mapping file
with open("final_basel_performance.json", "w") as outfile:
    json.dump(perform_dict, outfile)