# give some time reference to the user
print('Importing Gradio app packages... (first launch takes about 3-5 minutes)')

import gradio as gr
import yaml
import skimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import seaborn as sns

from cytof import classes
from classes import CytofImage, CytofCohort, CytofImageTiff
from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import show_color_table

OUTDIR = './output'

def cytof_tiff_eval(file_path, marker_path, cytof_state):
    # set to generic names because uploaded filenames is unpredictable
    slide = 'slide0'
    roi = 'roi1'

    # read in the data
    cytof_img, _ = cytof_read_data_roi(file_path, slide, roi)

    # case 1. user uploaded TXT/CSV
    if marker_path is None:
        # get markers
        cytof_img.get_markers()

        # prepsocess
        cytof_img.preprocess()
        cytof_img.get_image()

    # case 2. user uploaded TIFF
    else:
        labels_markers = yaml.load(open(marker_path, "rb"), Loader=yaml.Loader)
        cytof_img.set_markers(**labels_markers)

    viz = cytof_img.check_channels(ncols=3, savedir='.')

    msg = f'Your uploaded TIFF has {len(cytof_img.markers)} markers'
    cytof_state = cytof_img

    return msg, viz, cytof_state


def channel_select(cytof_img):
    # one for define unwanted channels, one for defining nuclei, one for defining membrane
    return gr.Dropdown(choices=cytof_img.channels, multiselect=True), gr.Dropdown(choices=cytof_img.channels, multiselect=True), gr.Dropdown(choices=cytof_img.channels, multiselect=True)

def nuclei_select(cytof_img):
    # one for defining nuclei, one for defining membrane
    return gr.Dropdown(choices=cytof_img.channels, multiselect=True), gr.Dropdown(choices=cytof_img.channels, multiselect=True)

def modify_channels(cytof_img, unwanted_channels, nuc_channels, mem_channels):
    """
    3-step function. 1) removes unwanted channels, 2) define nuclei channels, 3) define membrane channels
    """
    
    cytof_img_updated = cytof_img.copy()
    cytof_img_updated.remove_special_channels(unwanted_channels)

    # define and remove nuclei channels
    nuclei_define = {'nuclei' : nuc_channels}
    channels_rm = cytof_img_updated.define_special_channels(nuclei_define)
    cytof_img_updated.remove_special_channels(channels_rm)

    # define and keep membrane channels
    membrane_define = {'membrane' : mem_channels}
    cytof_img_updated.define_special_channels(membrane_define)

    # only get image when need to derive from df. CytofImageTIFF has inherent image attribute
    if type(cytof_img_updated) is CytofImage:
        cytof_img_updated.get_image()

    nuclei_channel_str = ', '.join(channels_rm)
    membrane_channel_str = ', '.join(mem_channels)
    msg = 'Your remaining channels are: ' + ', '.join(cytof_img_updated.channels) + '.\n\n Nuclei channels: ' + nuclei_channel_str + '.\n\n Membrane channels: ' + membrane_channel_str
    return msg, cytof_img_updated

def update_dropdown_options(cytof_img, selected_self, selected_other1, selected_other2):
    """
    Remove the selected option in the dropdown from the other two dropdowns
    """
    updated_choices = cytof_img.channels.copy()
    unavail_options = selected_self + selected_other1 + selected_other2
    for opt in unavail_options:
        updated_choices.remove(opt)

    return gr.Dropdown(choices=updated_choices+selected_other1, value=selected_other1, multiselect=True), gr.Dropdown(choices=updated_choices+selected_other2, value=selected_other2, multiselect=True)


def cell_seg(cytof_img, radius):

    # check if membrane channel available
    use_membrane = 'membrane' in cytof_img.channels
    nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=use_membrane, radius=radius, show_process=False)
    
    # visualize nuclei and cells segmentation
    marked_image_nuclei = cytof_img.visualize_seg(segtype="nuclei", show=False)
    marked_image_cell = cytof_img.visualize_seg(segtype="cell", show=False)

    # visualizing nuclei and/or membrane, plus the first marker in channels
    marker_visualized = cytof_img.channels[0]
    
    # similar to plt.imshow()
    fig = px.imshow(marked_image_cell)

    # add scatter plot dots as legends
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='white'), name='membrane boundaries'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='yellow'), name='nucleus boundaries'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red'), name='nucleus'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green'), name=marker_visualized))
    fig.update_layout(legend=dict(orientation="v", bgcolor='lightgray'))

    return fig, cytof_img

def feature_extraction(cytof_img, cohort_state, percentile_threshold):
    
    # extract and normalize all features
    cytof_img.extract_features(filename=cytof_img.filename)
    cytof_img.feature_quantile_normalization(qs=[percentile_threshold]) 

    # create dir if not exist
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    cytof_img.export_feature(f"df_feature_{percentile_threshold}normed", os.path.join(OUTDIR, f"feature_{percentile_threshold}normed.csv"))
    df_feature = getattr(cytof_img, f"df_feature_{percentile_threshold}normed" )

    # each file upload in Gradio will always have the same filename
    # also the temp path created by Gradio is too long to be visually satisfying.
    df_feature = df_feature.loc[:, df_feature.columns != 'filename']
    
    # calculates quantiles between each marker and cell
    cytof_img.calculate_quantiles(qs=[75])

    dict_cytof_img = {f"{cytof_img.slide}_{cytof_img.roi}": cytof_img}
    
    # convert to cohort and prepare downstream analysis
    cytof_cohort = CytofCohort(cytof_images=dict_cytof_img, dir_out=OUTDIR)
    cytof_cohort.batch_process_feature()
    cytof_cohort.generate_summary()

    cohort_state = cytof_cohort

    msg = 'Feature extraction completed!'
    return cytof_img, cytof_cohort, df_feature

def co_expression(cytof_img, percentile_threshold):
    feat_name = f"{percentile_threshold}normed"
    df_co_pos_prob, df_expected_prob = cytof_img.roi_co_expression(feature_name=feat_name, accumul_type='sum', return_components=False)
    epsilon = 1e-6 # avoid divide by 0 or log(0)

    # Normalize and fix Nan
    edge_percentage_norm = np.log10(df_co_pos_prob.values / (df_expected_prob.values+epsilon) + epsilon)
    
    # if observed/expected = 0, then log odds ratio will have log10(epsilon)
    # no observed means co-expression cannot be determined, does not mean strong negative co-expression
    edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0
    
    # do some post processing
    marker_all_clean = [m.replace('_cell_sum', '') for m in df_expected_prob.columns]
    
    # fig = plt.figure()
    clustergrid = sns.clustermap(edge_percentage_norm,
                    # clustergrid = sns.clustermap(edge_percentage_norm,
                    center=np.log10(1 + epsilon), cmap='RdBu_r', vmin=-1, vmax=3,
                    xticklabels=marker_all_clean, yticklabels=marker_all_clean)
    
    # retrieve matplotlib.Figure object from clustermap
    fig = clustergrid.ax_heatmap.get_figure()

    return fig, cytof_img

def spatial_interaction(cytof_img, percentile_threshold, method, cluster_threshold):
    feat_name = f"{percentile_threshold}normed"

    df_expected_prob, df_cell_interaction_prob = cytof_img.roi_interaction_graphs(feature_name=feat_name, accumul_type='sum', method=method, threshold=cluster_threshold)
    epsilon = 1e-6

    # Normalize and fix Nan
    edge_percentage_norm = np.log10(df_cell_interaction_prob.values / (df_expected_prob.values+epsilon) + epsilon)

    # if observed/expected = 0, then log odds ratio will have log10(epsilon)
    # no observed means interaction cannot be determined, does not mean strong negative interaction
    edge_percentage_norm[edge_percentage_norm == np.log10(epsilon)] = 0
    
    # do some post processing
    marker_all_clean = [m.replace('_cell_sum', '') for m in df_expected_prob.columns]
    

    clustergrid = sns.clustermap(edge_percentage_norm,
                                # clustergrid = sns.clustermap(edge_percentage_norm,
                                center=np.log10(1 + epsilon), cmap='bwr', vmin=-2, vmax=2,
                                xticklabels=marker_all_clean, yticklabels=marker_all_clean)

    # retrieve matplotlib.Figure object from clustermap
    fig = clustergrid.ax_heatmap.get_figure()

    return fig, cytof_img

def get_marker_pos_options(cytof_img):
    options = cytof_img.channels.copy()

    # nuclei is guaranteed to exist after defining channels
    options.remove('nuclei')

    # search for channel "membrane" and delete, skip if cannot find
    try:
        options.remove('membrane')
    except ValueError:
        pass

    return gr.Dropdown(choices=options, interactive=True), gr.Dropdown(choices=options, interactive=True) 

def viz_pos_marker_pair(cytof_img, marker1, marker2, percentile_threshold):
  
    stain_nuclei1, stain_cell1, color_dict = cytof_img.visualize_marker_positive(
                                        marker=marker1,
                                        feature_type="normed",
                                        accumul_type="sum",
                                        normq=percentile_threshold,
                                        show_boundary=True,
                                        color_list=[(0,0,1), (0,1,0)], # negative, positive
                                        color_bound=(0,0,0),
                                        show_colortable=False)

    stain_nuclei2, stain_cell2, color_dict = cytof_img.visualize_marker_positive(
                                        marker=marker2,
                                        feature_type="normed",
                                        accumul_type="sum",
                                        normq=percentile_threshold,
                                        show_boundary=True,
                                        color_list=[(0,0,1), (0,1,0)], # negative, positive
                                        color_bound=(0,0,0),
                                        show_colortable=False)
  
    # create two subplots
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=(f"positive {marker1} cells", f"positive {marker2} cells"))
    fig.add_trace(px.imshow(stain_cell1).data[0], row=1, col=1)
    fig.add_trace(px.imshow(stain_cell2).data[0], row=1, col=2)

    # Synchronize axes
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    fig.update_layout(title_text=" ")

    return fig

def phenograph(cytof_cohort):
    key_pheno = cytof_cohort.clustering_phenograph()

    df_feats, commus, cluster_protein_exps, figs, figs_scatter, figs_exps = cytof_cohort.vis_phenograph(
        key_pheno=key_pheno,
        level="cohort",
        save_vis=False,
        show_plots=False,
        plot_together=False)

    umap = figs_scatter['cohort']
    expression = figs_exps['cohort']['cell_sum']

    return umap, cytof_cohort

def cluster_interaction_fn(cytof_img, cytof_cohort):
    # avoid calling the clustering algorithm again. cohort is guaranteed to have one phenogrpah
    key_pheno = list(cytof_cohort.phenograph.keys())[0]
    
    epsilon = 1e-6
    interacts, clustergrid = cytof_cohort.cluster_interaction_analysis(key_pheno)
    interact = interacts[cytof_img.slide]
    clustergrid_interaction = sns.clustermap(interact, center=np.log10(1+epsilon),
                                cmap='RdBu_r', vmin=-1, vmax=1,
                                xticklabels=np.arange(interact.shape[0]),
                                yticklabels=np.arange(interact.shape[0]))

    # retrieve matplotlib.Figure object from clustermap
    fig = clustergrid.ax_heatmap.get_figure()

    return fig, cytof_img, cytof_cohort

def get_cluster_pos_options(cytof_img):
    options = cytof_img.channels.copy()

    # nuclei is guaranteed to exist after defining channels
    options.remove('nuclei')

    # search for channel "membrane" and delete, skip if cannot find
    try:
        options.remove('membrane')
    except ValueError:
        pass

    return gr.Dropdown(choices=options, interactive=True)

def viz_cluster_positive(marker, percentile_threshold, cytof_img, cytof_cohort):

    # avoid calling the clustering algorithm again. cohort is guaranteed to have one phenogrpah
    key_pheno = list(cytof_cohort.phenograph.keys())[0]
   
    # marker positive cell
    stain_nuclei1, stain_cell1, color_dict = cytof_img.visualize_marker_positive(
                                        marker=marker,
                                        feature_type="normed",
                                        accumul_type="sum",
                                        normq=percentile_threshold,
                                        show_boundary=True,
                                        color_list=[(0,0,1), (0,1,0)], # negative, positive
                                        color_bound=(0,0,0),
                                        show_colortable=False)

    # attch PhenoGraph results to individual ROIs
    cytof_cohort.attach_individual_roi_pheno(key_pheno, override=True)

    # PhenoGraph clustering visualization
    pheno_stain_nuclei, pheno_stain_cell, color_dict = cytof_img.visualize_pheno(key_pheno=key_pheno)

    # create two subplots
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, subplot_titles=(f"positive {marker} cells", "PhenoGraph clusters on cells"))
    fig.add_trace(px.imshow(stain_cell1).data[0], row=1, col=1)
    fig.add_trace(px.imshow(pheno_stain_cell).data[0], row=1, col=2)

    # Synchronize axes
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    fig.update_layout(title_text=" ")

    return fig, cytof_img, cytof_cohort

# Gradio App template
with gr.Blocks() as demo:
    cytof_state = gr.State(CytofImage())

     # used in scenrios where users define/remove channels multiple times
    cytof_original_state = gr.State(CytofImage())

    gr.Markdown("# Step 1. Upload images")
    gr.Markdown('You may upload one or two files depending on your use case.')
    gr.Markdown('Case 1: A single TXT or CSV file that contains information about antibodies, rare heavy metal isotopes, and image channel names. Make sure files are following the CyTOF, IMC, or multiplex data convention. Leave the `Marker File` upload section blank.')
    gr.Markdown('Case 2: Multiple file uploads required. First, a TIFF file containing Regions of Interest (ROIs) stored as multiplexed images. Then, upload a `Marker File` listing the channels to identify the antibodies.')

    with gr.Row(): # first row where 1) asks for TIFF upload and 2) displays marker info
        img_path = gr.File(file_types=[".tiff", '.tif', '.txt', '.csv'], label='(Required) A file containing Regions of Interest (ROIs) of multiplexed imaging slides.')
        img_info = gr.Textbox(label='Marker information', info='Ensure the number of markers displayed below matches the expected number.')

    with gr.Row(equal_height=True): # second row where 1) asks for marker file upload and 2) displays the visualization of individual channels
        with gr.Column():
            marker_path = gr.File(file_types=['.txt'], label='(Optional) Marker File. A list used to identify the antibodies in each TIFF layer. Upload one TXT file.')
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Upload")
        img_viz = gr.Plot(label="Visualization of individual channels")
    
    gr.Markdown("# Step 2. Modify existing channels")
    gr.Markdown("After visualizing the individual channels, did you notice any that should not be included in the next steps? Remove those if so.")
    gr.Markdown("Define channels designed to visualize nuclei. Optionally, define channels designed to visualize membranes.")

    with gr.Row(equal_height=True): # third row selects nuclei channels
        with gr.Column():
            selected_unwanted_channel = gr.Dropdown(label='(Optional) Select the unwanted channel', interactive=True)
            selected_nuclei = gr.Dropdown(label='(Required) Select the nuclei channel', interactive=True)
            selected_membrane = gr.Dropdown(label='(Optional) Select the membrane channel', interactive=True)
            
            define_btn = gr.Button('Modify channels')

        channel_feedback = gr.Textbox(label='Channels info update')

        # upload the file, and gather channel info. Then populate to the unwanted_channel, nuclei, and membrane components
        submit_btn.click(
            fn=cytof_tiff_eval, inputs=[img_path, marker_path, cytof_original_state], outputs=[img_info, img_viz, cytof_original_state], 
            api_name='upload'
        ).success(
            fn=channel_select, inputs=cytof_original_state, outputs=[selected_unwanted_channel, selected_nuclei, selected_membrane]
        )

    selected_unwanted_channel.change(fn=update_dropdown_options, inputs=[cytof_original_state, selected_unwanted_channel, selected_nuclei, selected_membrane], outputs=[selected_nuclei, selected_membrane], api_name='dropdown_monitor1') # api_name used to identify in the endpoints
    selected_nuclei.change(fn=update_dropdown_options, inputs=[cytof_original_state, selected_nuclei, selected_membrane, selected_unwanted_channel], outputs=[selected_membrane, selected_unwanted_channel], api_name='dropdown_monitor2')
    selected_membrane.change(fn=update_dropdown_options, inputs=[cytof_original_state, selected_membrane, selected_nuclei, selected_unwanted_channel], outputs=[selected_nuclei, selected_unwanted_channel], api_name='dropdown_monitor3')

    # modifies the channels per user input
    define_btn.click(fn=modify_channels, inputs=[cytof_original_state, selected_unwanted_channel, selected_nuclei, selected_membrane], outputs=[channel_feedback, cytof_state])

    gr.Markdown('# Step 3. Perform cell segmentation based on the defined nuclei and membrane channels')

    with gr.Row(): # This row defines cell radius and performs segmentation
        with gr.Column():
            cell_radius = gr.Number(value=5, precision=0, label='Cell size', info='Please enter the desired radius for cell segmentation (in pixels; default value: 5)')
            seg_btn = gr.Button("Segment")
        seg_viz = gr.Plot(label="Visualization of the segmentation. Hover over graph to zoom, pan, save, etc.")
        seg_btn.click(fn=cell_seg, inputs=[cytof_state, cell_radius], outputs=[seg_viz, cytof_state])
            
    gr.Markdown('# Step 4. Extract cell features')

    cohort_state = gr.State(CytofCohort())
    with gr.Row(): # feature extraction related functinos
        with gr.Column():
            gr.CheckboxGroup(choices=['Yes', 'Yes', 'Yes'], label='Note: This step will take significantly longer than the previous ones. A 130MB IMC file takes about 14 minutes to compute. Did you read this note?')
            norm_percentile = gr.Slider(minimum=50, maximum=99, step=1, value=75, interactive=True, label='Normalized quantification percentile')
            extract_btn = gr.Button('Extract')
        feat_df = gr.DataFrame(headers=['id','coordinate_x','coordinate_y','area_nuclei'], label='Feature extraction summary')
    
        extract_btn.click(fn=feature_extraction, inputs=[cytof_state, cohort_state, norm_percentile], 
        outputs=[cytof_state, cohort_state, feat_df])
    
    gr.Markdown('# Step 5. Downstream analysis')

    with gr.Row(): # show co-expression and spatial analysis
        with gr.Column():
            co_exp_viz = gr.Plot(label="Visualization of cell coexpression of markers")
            co_exp_btn = gr.Button('Run co-expression analysis')

        with gr.Column():
            spatial_viz = gr.Plot(label="Visualization of cell spatial interaction of markers")
            cluster_method = gr.Radio(label='Select the clustering method', value='k-neighbor',  choices=['k-neighbor', 'distance'], info='K-neighbor: classifies the threshold number of surrounding cells as neighborhood pairs. Distance: classifies cells within threshold distance as neighborhood pairs.')
            cluster_threshold = gr.Slider(minimum=1, maximum=100, step=1, value=30, interactive=True, label='Clustering threshold')

            spatial_btn = gr.Button('Run spatial interaction analysis')

        co_exp_btn.click(fn=co_expression, inputs=[cytof_state, norm_percentile], outputs=[co_exp_viz, cytof_state])
        # spatial_btn logic is in step6. This is populate the marker positive dropdown options

    gr.Markdown('# Step 6. Visualize positive markers')
    gr.Markdown('Select two markers for side-by-side comparison to visualize their positive states in cells. This serves two purposes. 1) Validate the co-expression analysis results. High expression level should mean a similar number of positive markers within the two slides, whereas low expression level mean a large difference of in the number of positive markers. 2) Validate the spatial interaction analysis results. High interaction means the two positive markers are in close proximity of each other (proximity is previously defined in `clustering threshold`), and vice versa.')
    
    with gr.Row(): # two marker positive visualization - dropdown options
        selected_marker1 = gr.Dropdown(label='Select one marker', info='Select a marker to visualize', interactive=True)
        selected_marker2 = gr.Dropdown(label='Select another marker', info='Selecting the same marker as the previous one is allowed', interactive=True)
        pos_viz_btn = gr.Button('Visualize these two markers')

        
    with gr.Row(): # two marker positive visualization - visualization
        marker_pos_viz = gr.Plot(label="Visualization of the two markers. Hover over graph to zoom, pan, save, etc.")
    
        spatial_btn.click(
            fn=spatial_interaction, inputs=[cytof_state, norm_percentile, cluster_method, cluster_threshold], outputs=[spatial_viz, cytof_state]
        ).success(
            fn=get_marker_pos_options, inputs=[cytof_state], outputs=[selected_marker1, selected_marker2]
        )
        pos_viz_btn.click(fn=viz_pos_marker_pair, inputs=[cytof_state, selected_marker1, selected_marker2, norm_percentile], outputs=[marker_pos_viz])


    gr.Markdown('# Step 7. Phenogrpah Clustering')
    gr.Markdown('Cells can be clustered into sub-groups based on the extracted single-cell data. Time reference: a 300MB IMC file takes about 2 minutes to compute.')

    with gr.Row(): # add two plots to visualize phenograph results
        phenograph_umap = gr.Plot(label="UMAP results")
        cluster_interaction = gr.Plot(label="Spatial interaction of clusters")

    
    with gr.Row(equal_height=False): # action components
        umap_btn = gr.Button('Run Phenograph clustering')
        cluster_interact_btn = gr.Button('Run clustering interaction')
        cluster_interact_btn.click(cluster_interaction_fn, inputs=[cytof_state, cohort_state], outputs=[cluster_interaction, cytof_state, cohort_state])

    with gr.Row():
        with gr.Column():
            selected_cluster_marker = gr.Dropdown(label='Select one marker', info='Select a marker to visualize', interactive=True)
            cluster_positive_btn = gr.Button('Compare clusters and positive markers')

        with gr.Column():
            cluster_v_positive = gr.Plot(label="Cluster assignment vs. positive cells. Hover over graph to zoom, pan, save, etc.")


        umap_btn.click(
            fn=phenograph, inputs=[cohort_state], outputs=[phenograph_umap, cohort_state]
        ).success(
            fn=get_cluster_pos_options, inputs=[cytof_state], outputs=[selected_cluster_marker], api_name='selectClusterMarker'
        )
        cluster_positive_btn.click(fn=viz_cluster_positive, inputs=[selected_cluster_marker, norm_percentile, cytof_state, cohort_state], outputs=[cluster_v_positive, cytof_state, cohort_state])


    # clear everything if clicked
    clear_components = [img_path, marker_path, img_info, img_viz, channel_feedback, seg_viz, feat_df, co_exp_viz, spatial_viz, marker_pos_viz, phenograph_umap, cluster_interaction, cluster_v_positive]
    clear_btn.click(lambda: [None]*len(clear_components), outputs=clear_components)


if __name__ == "__main__":
    demo.launch()

