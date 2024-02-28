# PRISM

PRISM (**P**rofiling of **R**NA **I**n-situ through **S**ingle-round i**M**aging) is an innovative method that employs an integer vector code to distinguish a wide array of RNA transcripts in large-scale tissues with sub-micron resolution through a single staining and imaging cycle, which make it fast and free of prblems of traditional methods like shifting and alignment.

For more information, please read the article. 

# Code Preview

Code for PRISM consists of the following parts: **probe_designer**, **image_process**, **gene_calling** **cell_segmentation**, **analysis_cell_typing** and **analysis_suncellular**. Data will be processed in this order.

# Data Architecture

Raw data base directory and processed data output directory can be whatever place you need. But its subdirectory should be like this:

```shell
Raw data root                           
├─RUN_ID1
├─RUN_ID2
├─...
├─RUN_IDN
```

```shell
Output root
├─RUN_ID1_processed(automate create)
│  ├─focal_stacked(automate create)
│  ├─background_corrected(automate create)
│  ├─registered(automate create)
│  ├─stitched(automate create)
│  ├─segmented(automate create)
│  ├─readout(automate create)
│  ├─analysis_cell_typing(automate create)
│  └─analysis_subcellular(automate create)
├─RUN_ID2_processed(automate create)
├─...
├─RUN_IDN_processed(automate create)
```
Your raw data should be in folder RUN_ID.

# Quick start

## Environment

This code should be run under `python 3.8`. Later version will bring some environment problems.

To run this code, packages must be installed with command:

```shell
pip install -r requirements.txt
```

And MATLAB engine should be installed, according to your local MATLAB version, you can follow [official guideline](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).  

For example, MATLAB R2021b：`python -m pip install matlabengine==9.11.21`

And **pip setuptools** needs a correct version. 

For MATLAB R2021b: `pip install --upgrade pip setuptools==57.5.0`

**Remind!** There are lots of paths or directories need to be edited in files mentioned below.


## Probe Design
This step is not necessary all the time because you can desigh the probe yourself or contact us for help. But if you want to design the probe easily by yourself, see: https://github.com/Tangmc-kawa/probe_designer


## Image Process 2D
Step 1 and Step 2 is used to generate a complete image for each channel used in experiment. If you have other methods to per form this image process, store the name of each channel's image as `cyc_1_channel.tif` 
### Step 1: Scan_fstack
Edit the directory in python file `scan_fstack.py` as the directory you wish. Run the code: 
```shell
python scan_fstack.py Raw_data_root
```
**Remark**: This step is aimed to process raw images captured in small field and multi channel. you can use it to process your own experiment data. We have provided a preprocessed example data for Step 2 and pipeline after, which is located at `./dataset/processed/_example_dataset_processed`.  You can change the RUN_ID in each script to `_example_dataset` and continue the following steps.

### Step 2: Image_process
Edit the directory in python file `image_process_after_stack.py` the same as the directory before. Run the code: 
```shell
python image_proess_after_stack.py
```
**Remark**: This step includes register the subimages, correct the background and stitch them to a whole image, Results will be stitched n (n is the number of channels you use) big images in `RUN_IDx_processed/stitched/`, which will be used in next part.

### Step 3: Multi_channel_readout
Edit the directory in python file `multi_channel_readout.py` the same as the directory before. Run the code: 
```shell
python multi_channel_readout.py
```
**Remark**: Using big images generated in the previous step. Signal spots and their intensity can be extract using `multi_channel_readout.py`. It will generate two csv files named `intensity_all.csv` `intensity_all_deduplicated.csv` in the directory `RUN_IDx_processed/readout/`.

## Image Process 3D


## Gene Calling
Edit the directory in `Gene_calling.py` and run it. A csv file containing spots coordinate and gene name will be generated, named `output_root/RUN_IDx_processed/readout/mapped_genes.csv`.

## Cell Segmentation
Edit the directory in `segment.py` and run it. This code will segment cell nucleus according to DAPI channel. A csv file containing the coordinate of nucleus centroid will be generated named `output_root/whatever_name_processed/segmented/centroids_all.csv`.

## Expression Matrix
Edit the directory and run the jupyter notebook file `integrated_analysis_SPRINTseq.ipynb`, important post processing results will be shown inside the jupyter.

The provided dataset is just an example data, so we provided another true experimental data for post-analysis in folder `example_dataset_whole_brain`. Run `merge.py` to build dataset from splited dataset, and replace `mapped_genes.csv` and `centroids_all.csv` mentioned above with file in this folder to use this dataset for Matrix Analysis.

## Cell typing and related analysis


## Subcellular analysis

