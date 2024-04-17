# PRISM

PRISM (**P**rofiling of **R**NA **I**n-situ through **S**ingle-round i**M**aging) is an innovative method that employs an radius vector code to distinguish a wide array of RNA transcripts in large-scale tissues with sub-micron resolution through a single staining and imaging cycle, which make it fast and free of prblems of traditional methods like shifting and alignment.

For more information, please read the article.

# Code Preview

Code for PRISM consists of the following parts: **probe_designer**, **image_process**, **gene_calling** **cell_segmentation**, **analysis_cell_typing** and **analysis_suncellular**. Data will be processed in this order.

# Data Architecture

Raw data base directory and processed data output directory can be whatever place you need. But its subdirectory should be like this:

```shell
Raw data root
├─RUN_ID1
│  ├─cyc1
│  │  ├─C001-T0001-cy3-Z000.tif
│  │  ├─C001-T0001-cy3-Z001.tif
│  │  ├─...
│  │  ├─C001-T0004-FAM-Z006.tif
│  │  ├─...
│  │  └─C001-T0108-TxRed-Z008.tif
│  └─cyc2(usually not exist in PRISM)
├─RUN_ID2
├─...
└─RUN_IDN
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
│  ├─cidre(used in 3D spot extraction, automate create)
│  ├─airlocalize_stack(used in 3D spot extraction, automate create)
│  ├─analysis_cell_typing(automate create)
│  └─analysis_subcellular(automate create)
├─RUN_ID2_processed(automate create)
├─...
└─RUN_IDN_processed(automate create)
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

cid_dir = dest_dir / 'cidre'                # pipeline_3D.py
air_dir = dest_dir / 'airlocalize_stack'    # pipeline_3D.py

# In following analysis
src_dir = BASE_DIR / f'{RUN_ID}_processed'  # processed data
stc_dir = src_dir / 'stitched'              # image_process_after_stack.py
read_dir = src_dir / 'readout'              # multi_channel_readout.py
seg_dir = src_dir / 'segmented'             # segment2D.py or segment3D.py or expression_matrix.py
visual_dir = src_dir / 'visualization'      # folder for figures...
```

## How to get our data

raw data are provided in [spatial_transcriptome_raw_data](https://github.com/huanglab111/spatial_transcriptome_raw_data/), download from it based on your need.

Some raw data for our articles are uploaded as:

- HCC2D: `spatial_transcriptome_raw_data\20230523_HCC_PRISM_probe_refined`

For more raw data, contact us from huanglab111@gmail.com.

# Pipeline

The pipeline can be explained as:

```shell
+-------------+     +------------+     +---------------+     +------------------+     +--------------+     +-----------------+
| ProbeDesign | --> | Experiment | --> | 2D Data Stack | --> | 2D Image Process | --> | Gene Calling | --> | 2D Cell Segment |
+-------------+     +------------+     +---------------+     +------------------+     +--------------+     +-----------------+
                        or|                 or|                                                                 |
                          v                   v                                                                 v
                      +---------+     +------------------+     +--------------+     +-----------------+    +----------+
                      | 3D Data | --> | 3D Image Process | --> | Gene Calling | --> | 3D Cell Segment |--> | Analysis |
                      +---------+     +------------------+     +--------------+     +-----------------+    +----------+
```

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

### 3D tif files as input

If your images are captured by confocal, lightsheet or any other 3D microscopy and you have a registered and stitched grayscale 3d image of each channel in tiff format. Spot extraction can be performed using airlocalize.py with proper parameters.

```shell
python image_process/lib/AIRLOCALIZE/airlocalize.py
```

**Remarks**:

- The default image axis is 'XYZ' in airlozalize, if you need other axis order like 'ZXY', please use np.transpose() in previous step or modify the 'self.retrieve_img()' function in `Image_process/lib/AIRLOCALIZE/airlocalize_data.py`.

- AIRLOCALIZE was first written by . It includes predetection signal spots on a feature image(like DoG or LoG or Gaussian smoothed) and fit the accurate location and intensity of spots on original image. Important parameters includes:

```
# scaling
scale: scale the image to drop too low or too high spots.
scaleMaxRatio: the final absolute upperbound of scaled image, don't set it 1 to prevent excess of np.uint16 and cause errors.

# predetection
featureExtract: feature image method, LoG is thought best while it takes relatively high calculation resources. DoG with properly set filterLo and filterHi can be seen as a similar replacement.
ThreshLevel: list, different channel can be set different. The higher, the less points detected.
maxSpots: predetected spots number, got by strip the first 'maxSpots' spots sorted by intensity from high to low.

# Gaussian fit
Psfsigma: the sigma of your gaussian like signal points, determined by your real data.
```

### 3D reconstructure of 2D images

If your images are captured as mentioned in [Data Architecture](#Data-Architecture) above and you want to restore the z stack infomation(even if only 10um), change the parameters file path in `pipeline_3D.py` and run:

```shell
python pipeline_3D.py
```

to read the intensity from raw images.

**Remark**: This pipeline includes 2D process as cycle shift and global position of each tile is needed from 2D pipeline. After that, airlocalize is performed to extract spots in 3d (z stack number as the depth). Remember to change the parameters file path and adjust the parameters for your own data before you run the code.

---

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

or

```shell
python cell_segmentation/segment3D.py
```

This code will segment cell nucleus according to DAPI channel. A csv file containing the coordinate of nucleus centroid will be generated in`seg_dir` as `centroids_all.csv`.

Edit the directory in python file `gene_calling/expression_matrix.py`, and run:

```shell
python cell_segmentation/expression_matrix.py
```

the expression matrix will be generated in `seg_dir` as `expression_matrix.csv`

**Remarks**:

- `Segmentation3D.py` needs stardist environment as it use trained network to predict the shape and centroid of nucleus in 3D. For more information, see: https://github.com/stardist/stardist.
- Our strategy to generate expression matris in general assign rna to its nearest centroid of cell nucleus (predicted by dapi) so it requires `centroids.csv(dapi_predict.csv)` of cell nucleus and `mapped_genes.csv` generated in previous steps. If you have other strategies which performed better in your data, you can replace this step with it.

---

# Cell typing and analysis

Cell typing analysis were performed based on the feature of each sample and may differ slightly, the analysis code for each sample is located in corresponding folders.

**Remark**: If you want to run harmony or other scRNA seq data based analysis, download the single cell data from NCBI and CNP and store them in this structure:

```shell
dataset_sc_rnaseq
├─sc_data_HCC
│  ├─processed_CNP0000650_CD45-
│  │      HCC_cell_metadata.txt
│  │      HCC_cell_metadata.txt.md5
│  │      HCC_log_tpm_expression_matrix.txt
│  │      HCC_log_tpm_expression_matrix.txt.gz.md5
│  │
│  ├─processed_GSE140228_immune
│  │      GSE140228_cell_info_Smartseq2.tsv
│  │      GSE140228_gene_info_Smartseq2.tsv
│  │      GSE140228_read_counts_Smartseq2.csv
│  │
│  └─processed_GSE149614
│         GSE149614_HCC.scRNAseq.S71915.count.txt
│         GSE149614_HCC.scRNAseq.S71915.normalized.txt
│         HCC.metadata.txt
│
└─sc_data_mousebrain
   │  l1_cortex1.loom
   │  l1_cortex2.loom
   │  l1_cortex3.loom
   │  l1_hippocampus.loom
   │  l1_hypothalamus.loom
   │  l1_thalamus.loom
   └─cache
           l1_cortex1.h5ad
           l1_cortex2.h5ad
           l1_cortex3.h5ad
           l1_hippocampus.h5ad
```

## PRISM2D mousebrain

You can get the cell typing data and figures using `N_cluster_scanpy.py` in `dataset\processed\PRISM2D_mousebrain`. Cells were first devided into excitory neurons, inhibitory neurons and glia. And subtypes like L4 Ex neuron, In-Sst neuron and oligodendrocyte were futher identified in their major types.

## PRISM2D mouseEmbryo

26 Embryo slices of E12.5 to E14.5 are analyzed using `dataset/processed/PRISM2D_MouseEmbryo/Cell_typing_Embryo.ipynb`. Cell type is determined using gene with highest abundance in each cell. And spatial projection of each type can be seen here, too.

## PRISM HCC

For HCC, the data passed quality control was processed standard pipeline, including `normalization`, `log1p`, `regress_out`, and `scaling`, before performing Principal Component Analysis (PCA). This foundational work facilitated the integration of spatial transcriptomics data with three distinct single-cell transcriptome datasets (GSE151530, GSE140228, CNP0000650) using the Harmony algorithm within the Scanpy framework. Utilizing high-resolution Leiden shared nearest neighbor clustering. We identified 100 subclusters, which were meticulously annotated based on the expression of numerous housekeeping and marker genes associated with HCC, revealing 33 subclusters representing a diverse array of immune and nonimmune cell types. Notably, the challenge posed by the absence of well-defined marker genes for liver cells was addressed by leveraging elevated Hepatitis B Virus (HBV) expression as a proxy, allowing for the precise annotation of liver cell types. This innovative approach helped to reclassify cells initially tagged as other types, ultimately enriching the final dataset with 59,517 cells categorized into 17 major types and 34 subtypes. Each cell type was visually distinguished using a unique color scheme in the convex hull representation, with the dataset displayed in two dimensions using Uniform Manifold Approximation and Projection (UMAP).

Special attention was given to the issue of gene overfitting in the context of Harmony integration, where some genes, due to numerical constraints, formed peripheral clusters in the UMAP visualization. This was mitigated by applying threshold filtering based on the bimodal trough of marker gene histograms, ensuring cell types' precise identification and annotation. Moreover, the utilization of HBV content to annotate normal hepatocytes among cells initially classified as 'other' demonstrated a nuanced approach to cell type classification in the absence of conventional marker genes.

### PRISM2D HCC

HCC analysis of 2D slice is in `dataset/processed/PRISM2D_HCC/PRISM_HCC_2D_cell_typing_and_analysis.ipynb`.

### PRISM3D HCC

HCC analysis of Quasi-3D sample is in `dataset/processed/PRISM_HCC_of_20_slides/PRISM_HCC_3D_cell_typing_and_analysis.ipynb`

## PRISM3D mousebrain

The analysis of the 3D mouse brain data through spatial transcriptomics and single-cell transcriptomics is an intricate process that involves several critical steps to classify and annotate cell types accurately. This process begins with the integration of diverse datasets to ensure a unified and comprehensive view of the brain's cellular landscape.

Codes for different tissues are located at:

`dataset/processed/20230704_PRISM3D_mousebrain_CTX_rm_doublet/PRISM3D_cell_typing_and_analysis.ipynb`
`dataset/processed/20230705_PRISM3D_mousebrain_HT_rm_doublet/PRISM3D_cell_typing_and_analysis.ipynb`
`dataset/processed/20230706_PRISM3D_mousebrain_TH_rm_doublet/PRISM3D_cell_typing_and_analysis.ipynb`
`dataset/processed/20230710_PRISM3D_mousebrain_HP_rm_doublet/PRISM3D_cell_typing_and_analysis.ipynb`

### Data Integration and Harmony

The initial step in the analysis involves integrating spatial transcriptomics data from 3D mouse brain scans with single-cell transcriptomic data obtained from external databases like mousebrain.org. The Harmony algorithm plays a pivotal role here, as it aligns data from different sources, effectively reducing batch effects and discrepancies that typically arise from varied experimental conditions. Also, it makes our low count data better annotated with the help of scRNA data.

### High-Resolution Clustering

Following data integration, the next step is to apply the Leiden algorithm for clustering. This method detects communities of cells that exhibit similar gene expression profiles, thus grouping them into clusters. The Leiden algorithm is particularly adept at revealing fine-grained cellular distinctions, which identifies subtle differences among cell types.

### Tissue-Specific Analysis and Annotation

Each brain region—such as the cortex, hippocampus, thalamus, and hypothalamus—is analyzed separately to account for their unique cellular compositions and biological functions. This tissue-specific approach allows for more precise annotations, as each region may express different sets of marker genes indicative of specific cellular functions and states. Clusters are annotated based on the expression of these marker genes, which are carefully selected for their known associations with particular cell types and subtypes.

### Manual Refinement and Validation

Post-clustering, there is an essential phase of manual annotation where experts review the automated classifications. This manual oversight ensures that the computational predictions align with biological knowledge and empirical evidence. It's a critical step for validating the accuracy of the cell type assignments and for making necessary adjustments based on expert knowledge of brain biology.

### Visualization and Interpretation

The final clusters and annotations are visualized using techniques like UMAP, which simplifies the complex data into a two-dimensional space for easier interpretation. In this space, cells with similar types are plotted close to each other, and different colors represent different cell types, providing a clear visual distinction among the varied cell populations. This visual representation helps in the intuitive understanding of the data, highlighting the distribution and relationships of cell types across the brain.

Also, the mask of different cells are written in tif using this code and visualized using imaris.

### Comprehensive Documentation

Throughout the process, every step, from data preprocessing to final visualization, is documented in a Jupyter Notebook. This documentation includes the code used for analysis, detailed comments explaining each step, and the visual outputs such as graphs and UMAP plots. This not only ensures reproducibility and transparency but also serves as a valuable resource for further research and analysis.

Together, these steps form a robust framework for the detailed and nuanced analysis of cellular diversity within the mouse brain, providing insights that are crucial for understanding neurological functions and disorders.

# Subcellular analysis

Subcellular analysis is mainly performed on PRISM3D mousebrain dataset. In this study, we developed a comprehensive approach to analyze the subcellular distribution and polarity of RNA within cells, focusing on the variability across different tissues and cell types. Here, we outline the key methodologies and data analysis processes employed to dissect complex patterns of RNA behavior and its implications for cellular function.

See the details in `analysis_subcellular/PRISM3D_subcellular_analysis.ipynb`.

## Data Collection

We collected data on RNA molecules' positions relative to the nucleus and classified them based on four parameters:

- **Tissue Type**: Categorized into four groups (CTX, HP, TH, HY).
- **Cell Type**: Included excitatory neurons, inhibitory neurons, glial cells, and their subtypes.
- **Gene Type**: Focused on thirty detected genes.
- **Cellular Location**: Classified as inside or outside the nucleus.

## Statistical Analysis

Using the chi-square test, we assessed the relationship between RNA distribution and the categorical variables of tissue and cell type. This test helped determine if the observed distribution patterns were significantly different from what would be expected under a random distribution.

## Calculation of Roe (Ratio of Observed to Expected)

For a more nuanced analysis, we computed the Roe for RNA molecules in each category. This involved dividing the observed count of RNA molecules by the expected count, calculated based on overall data trends. The Roe helped us understand specific clustering patterns under different classification conditions.

## Dimensionality Reduction Techniques

To streamline the complexity inherent in our data, we applied dimensionality reduction in two key areas:

- **Gene Type Dimension**: By averaging the Roe values across all thirty genes, we reduced the complexity and highlighted the contributions of marker genes.
- **Nuclear/Extranuclear Dimension**: We computed the ratio of RNA outside the nucleus to that inside, summarizing the data into average relative abundances for different categories.

## Polarity Analysis

We quantified RNA polarity within cells by comparing each RNA's centroid distance from the nucleus centroid to the average distance of all RNAs to the nucleus. We established thresholds for high, medium, and low polarity, allowing us to categorize the RNA molecules accordingly and observe the variability in polarity distribution across different tissue and cell types.

## Simplifying Polarity Data

Similar to our approach with gene type, we reduced the complexity of polarity data by using a high-to-low polarity ratio. This step enabled us to achieve a clear, summarized view of RNA polarity distribution across different categories.

This structured approach allowed us to extract meaningful insights into RNA distribution and polarity within cells, enhancing our understanding of cellular dynamics and potentially informing future biological research and therapeutic strategies.
