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

Important results of each step are stored in corresponding directory under RUN_ID_processed. In this example, steps' dir are defined as:

```shell
# In image process
dest_dir = BASE_DIR / f'{RUN_ID}_processed' # processed data
aif_dir = dest_dir / 'focal_stacked'        # scan_fstack.py
sdc_dir = dest_dir / 'background_corrected' # image_process_after_stack.py
rgs_dir = dest_dir / 'registered'           # image_process_after_stack.py
stc_dir = dest_dir / 'stitched'             # image_process_after_stack.py
rsz_dir = dest_dir / 'resized'              # image_process_after_stack.py

# In following analysis
src_dir = BASE_DIR / f'{RUN_ID}_processed'  # processed data
stc_dir = src_dir / 'stitched'              # image_process_after_stack.py
read_dir = src_dir / 'readout'              # multi_channel_readout.py
seg_dir = src_dir / 'segmented'             # segment2D.py or segment3D.py or expression_matrix.py

```

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

## Probe Design

This step is not always necessary because you can design the probe with specific binding sites, barcodes and corresponding fluorophore probes manually or contact us for help. But if you want to design the probe easily or in bulk, see: https://github.com/Tangmc-kawa/probe_designer.

**\*Remind!** Lots of paths or directories need editing in files mentioned below.\*

---

## Image Process 2D

Step 1 and Step 2 is used to generate a complete image for each channel used in experiment. If you have other methods to per form this image process, store the name of each channel's image as `cyc_1_channel.tif`

### Step 1: Scan_fstack

Edit the directory in python file `scan_fstack.py` and run the code:

```shell
python image_process/scan_fstack.py Raw_data_root
```

**Remark**: This step is aimed to process raw images captured in small field and multi channel. you can use it to process your own experiment data. We have provided a preprocessed example data for Step 2 and pipeline after, which is located at `./dataset/processed/_example_dataset_processed`. You can change the RUN_ID in each script to `_example_dataset` and continue the following steps.

### Step 2: Image_process

Edit the directory in python file `image_process/image_process_after_stack.py` as the same, and run the code:

```shell
python image_process/image_proess_after_stack.py
```

**Remark**: This step includes register the subimages, correct the background and stitch them to a whole image. Results will be stitched n (n is the number of channels you use) big images in `stc_dir`, which will be used in next part.

### Step 3: Multi_channel_readout

Edit the directory in python file `image_process/multi_channel_readout.py` as the same, and run the code:

```shell
python image_process/multi_channel_readout.py
```

**Remark**: This step requires stitched big images generated in the previous step. Signal spots and their intensity can be extract using `image_process/multi_channel_readout.py`. It will generate two csv files named `intensity_all.csv` `intensity_all_deduplicated.csv` in the directory `RUN_IDx_processed/readout/`.

## Image Process 3D

## Gene Calling

Edit the directory in python file `gene_calling/gene_calling_GUI.py`, run the code:

```shell
python gene_calling/gene_calling_GUI.py
```

and follow the indications of each step. The result should be at `read_dir/mapped_genes.csv` as default, and you can choose another directory in the GUI.

**Remark**: Gene calling for PRISM is performed by a Gaussian Mixture, mannual select by masks and evaluation of the confidence of each spot. It's expected to run on a GUI because some steps need human knowledge of the experiments like how the chemical environment or FRET would affect the fluorophores. You can also use `gene_calling/PRISM_gene_calling_GMM.ipynb` for customization or use `gene_calling/gene_calling_manual.ipynb` to set the threshold for each gene manually.

For more detail, see https://github.com/Tangmc-kawa/PRISM_gene_calling.

## Cell Segmentation

Edit the directory in python file `cell_segmentation/segment2D.py` or `cell_segmentation/segment3D.py` and run:

```shell
python cell_segmentation/segment2D.py
```

This code will segment cell nucleus according to DAPI channel. A csv file containing the coordinate of nucleus centroid will be generated in`seg_dir` as `centroids_all.csv`.

Edit the directory in python file `gene_calling/expression_matrix.py`, and run:

```shell
python cell_segmentation/expression_matrix.py
```

the expression matrix will be generated in `seg_dir` as `expression_matrix.csv`

**Remark**:

- `Segmentation3D.py` needs stardist environment as it use trained network to predict the shape and centroid of nucleus in 3D. For more information, see: https://github.com/stardist/stardist.
- Our strategy to generate expression matris in general assign rna to its nearest centroid of cell nucleus (predicted by dapi) so it requires `centroids` of cell nucleus and `mapped genes` generated in previous steps. If you have other strategies which performed better in your data, you can replace this step with it.

## Cell typing and related analysis

## Subcellular analysis
