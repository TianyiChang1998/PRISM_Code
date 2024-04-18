#!/bin/python3

'''
Read csv file with scanpy, and perform latter analysis. 
normalization, scale refer to STARmap.
Perform clustering on the dataset: (1)(2) scanpy.
FOR EMBRYO.
'''

import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import use as muse
from umap.umap_ import simplicial_set_embedding, find_ab_params
import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path('/mnt/data/local_processed_data')
RUN_ID = '20230608_AD_PRISM_1x1'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
seg_dir = src_dir / 'segmented'
visual_dir = src_dir / 'visualization'
express_dir = visual_dir / 'expression'
visual_dir.mkdir(exist_ok=True)
express_dir.mkdir(exist_ok=True)


# ===============
# read files
# ===============
# -- hyper-parameter --
sc.settings.verbosity = 3
# sc.settings.set_figure_params(dpi=200, frameon=False, figsize=(4, 4), facecolor='white')	# dpi: the resolution in dots per inch
muse('Agg')	# not show figures on the server
random_seed = 15 # 62 for CTX old,c15 for new good
data_dir = (visual_dir/'expression_matrix_0704_HT.csv') #expression_matrix_0704_CTX.csv
results_dir = visual_dir / 'result'
results_dir.mkdir(parents=True, exist_ok=True)

# -- read file --
print('Reading files from', data_dir, '\n')
adata = sc.read(data_dir, delimiter=',', first_column_names=True, cache=False) # read csv file and convert it into AnnData object
adata.obs['Cell_Index'] = adata.obs_names.astype('str').to_list()
adata.obs['celltype'] = [None] * adata.n_obs
type_factor = 'all'  # cell factor: 1, 2, 3 or 'all'
print('Get AnnData:\n', adata, '\n')


# =============
# preprocess
# =============
# (1) computing average-highly-expressed genes
# sc.pl.highest_expr_genes(adata, n_top=20, show=False)
# plt.savefig(results_dir/'highest_expr_genes.pdf')

# (2) basic filtering: no need
sc.pp.filter_cells(adata, min_genes=1)
sc.pp.filter_cells(adata, min_counts=3)
sc.pp.filter_genes(adata, min_cells=10)
print('AnnData after basic filtering:\n', adata, '\n')


# subset
sc.pp.subsample(adata, fraction=0.5, random_state=random_seed)
print('AnnData after subsampling:\n', adata, '\n')

# (3) filtering low-quality cells according to mt or rp genes: no need
# `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
# Question: log1p = True or False?
sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)  # calculate many qc metrics based on parameters passed in
adata.obs['mean_counts'] = np.array(adata.X).sum(axis=1) / np.count_nonzero(adata.X, axis=1)
adata.var['mean_counts_only_expr'] = np.array(adata.X).sum(axis=0) / np.count_nonzero(adata.X, axis=0)
## violin plot
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'mean_counts'], jitter=0.4, multi_panel=True, show=False)
plt.savefig(results_dir/(type_factor+'_qc.pdf'))
## feature correlation
## filtering by slicing the AnnData object: no need
# n_genes = 3
# adata = adata[(adata.obs.total_counts <= 100) & ((adata.obs.n_genes_by_counts > n_genes) | (adata.obs.mean_counts_only_expr > n_genes)), :]
print('AnnData after second filtering:\n', adata, '\n')

# doublets removal: no need

# (4) normalize the data
## library-size correction
median_transcript = np.median(adata.X.sum(axis=1))  
sc.pp.normalize_total(adata, target_sum=median_transcript)	# there may be different total counts in different cell stages, thus whether using total counts to normalize remains question
## logarithmize the data: no need
sc.pp.log1p(adata)

# batch removal like harmony: no need

# (5) identify highly-variable genes: no need
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# (6) freezes the state of the AnnData object
adata.raw = adata
# adata = adata.raw.to_adata()  # get back to an AnnData

# remove cell cycle effect: no need

# (7) filtering: no need
# highly_var = np.array([False]*30)
# sub_genes = np.array([6, 8, 11, 12, 13, 14, 16, 17, 20, 21, 23, 24, 27, 28, 30]) - 1
# highly_var[sub_genes] = True
# adata.var['highly_variable'] = highly_var
# adata = adata[:, adata.var.highly_variable]

# (8) regressing out and re-scaling
## regressing out
sc.pp.regress_out(adata, ['total_counts'])
## scaling
sc.pp.scale(adata, max_value=10) # clip values exceeding standard deviation 10
# np.savetxt(results_dir/'scaled_data.csv', adata.X, delimiter=',')
print('AnnData preprocessed:\n', adata, '\n')


# =======================
# Downstream analysis
# =======================
# -- dimension reduction & clustering --
# linear dimension reduction
sc.tl.pca(adata, n_comps=10)
sc.pl.pca_variance_ratio(adata, log=True, show=False)	# to find the best PC number
plt.savefig(results_dir/(type_factor+'_pca_var_ratio.pdf'))

# batch removal like harmony: no need

# find neighbors
sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=20) #100

# Leiden: resolution is changeable
sc.tl.leiden(adata, resolution=1, random_state=random_seed)
print('After clustering\n', adata, '\n')

# non-linear dimension reduction
# scanpy.umap
sc.tl.umap(adata, random_state=random_seed)

# umap.Umap
# print('Performing UMAP.')
# n_neighbors = 50
# max_pc = 15
# coordinates = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.01).fit_transform(adata.obsm['X_pca'][:, :max_pc])
# adata.obsm['X_umap'] = coordinates
# del coordinates


'''
# simplicial_set_embedding
max_pc = 15
min_dist = 0.8
X_umap = simplicial_set_embedding(data=adata.obsm['X_pca'][:, :max_pc],
                                  graph=adata.obsp['connectivities'].tocoo(),
                                  n_components=2,
                                  initial_alpha=1.0,
                                  a=find_ab_params(1.0, min_dist)[0],
                                  b=find_ab_params(1.0, min_dist)[1],
                                  gamma=1.0,
                                  negative_sample_rate=5,
                                  n_epochs=200,
                                  init='spectral',
                                  random_state=check_random_state(random_seed),
                                  metric='euclidean',
                                  metric_kwds={},
                                  verbose=False,
                                  densmap=False,
                                  densmap_kwds={},
                                  output_dens=False
                                 )
adata.obsm['X_umap'] = X_umap[0]
del X_umap
'''


# plot leiden
sc.pl.pca(adata, color='leiden', use_raw=False, title='Leiden Results', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_pca_leiden.pdf'), dpi=200, bbox_inches='tight')

sc.pl.umap(adata, color='leiden', use_raw=False, title='Leiden Results', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_leiden.pdf'), dpi=200, bbox_inches='tight')

# -- plot gene expression --
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12','PRISM_13','PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22','PRISM_23', 'PRISM_24', 'PRISM_25','PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29','PRISM_30']
# sc.pl.dotplot(adata, marker_genes, groupby='leiden', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
sc.pl.dotplot(adata, marker_genes, groupby='leiden', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_leiden.pdf'))

adata_coor = pd.DataFrame(adata.obsm['X_umap'], columns=['Coor_X', 'Coor_Y'], index=adata.obs['Cell_Index'])
df = pd.concat([adata_coor['Coor_X'], adata_coor['Coor_Y'], adata.obs.Cell_Index, adata.obs.leiden], axis=1)
df.to_csv(results_dir/(type_factor+'_cellIndex.csv'))



'''
# =====================================
# for excitatory/inhibitory/other cell
# =====================================
## excitatory
type_factor = 'excitatory'
cell_cluster = ['2', '3', '4', '6', '8', '14']
print('\n\nPerforming dimension reduction on', type_factor, 'clusters:', cell_cluster, '\n')
ann = adata[adata.obs.leiden.isin(cell_cluster)]

# -- dimension reduction & clustering --
# linear dimension reduction
sc.tl.pca(ann, n_comps=10)
sc.pl.pca_variance_ratio(ann, log=True, show=False)     # to find the best PC number
plt.savefig(results_dir/(type_factor+'_pca_var_ratio.pdf'))

# find neighbors
sc.pp.neighbors(ann, use_rep='X_pca', n_neighbors=45)

# Leiden: resolution is changeable
sc.tl.leiden(ann, resolution=0.7, random_state=random_seed)
print('After clustering\n', ann, '\n')

sc.tl.umap(ann, random_state=random_seed)

# plot leiden
sc.pl.umap(ann, color='leiden', use_raw=False, title='Leiden Results', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_leiden.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by leiden
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='leiden', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_leiden.pdf'))


# ex_celltypes = {'0':'0-L2/3 Ex neuron', '1':'1-L4 Ex neuron', '2':'2-L5 Ex neuron', '3':'0-L2/3 Ex neuron', '4':'2-L5 Ex neuron', '5':'5-Th-Ex', '6':'Other Ex neuron', '7':'3-L6a Ex neuron', '8':'Other Ex neuron', '9':'Other Ex neuron', '10':'4-L6b Ex neuron'}
ex_celltypes = {'0':'0-L2/3 Ex neuron', '1':'1-L4 Ex neuron', '2':'2-L5 Ex neuron', '3':'0-L2/3 Ex neuron', '4':'2-L5 Ex neuron', '5':'5-Th-Ex', '6':'999-Others', '7':'3-L6a Ex neuron', '8':'999-Others', '9':'999-Others', '10':'4-L6b Ex neuron'}
ann.obs['celltype'] = (ann.obs['leiden'].map(ex_celltypes).astype('category'))
# adata.obs['celltype'] = ann.obs.celltype
df1 = ann.obs.celltype

# plot celltype
sc.pl.umap(ann, color='celltype', use_raw=False, title='Cell Type', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_celltype.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by celltype
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='celltype', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_celltype.pdf'))

ann_coor = pd.DataFrame(ann.obsm['X_umap'], columns=['Coor_X', 'Coor_Y'], index=ann.obs['Cell_Index'])
df = pd.concat([ann_coor['Coor_X'], ann_coor['Coor_Y'], ann.obs.Cell_Index, ann.obs.leiden, ann.obs.celltype], axis=1)
df.to_csv(results_dir/(type_factor+'_cellIndex.csv'))
del ann
del ann_coor



## inhibitory
type_factor = 'inhibitory'
cell_cluster = ['0', '10']
print('\n\nPerforming dimension reduction on', type_factor, 'clusters:', cell_cluster, '\n')
ann = adata[adata.obs.leiden.isin(cell_cluster)]

# -- dimension reduction & clustering --
# linear dimension reduction
sc.tl.pca(ann, n_comps=8)
sc.pl.pca_variance_ratio(ann, log=True, show=False)     # to find the best PC number
plt.savefig(results_dir/(type_factor+'_pca_var_ratio.pdf'))

# find neighbors
sc.pp.neighbors(ann, use_rep='X_pca', n_neighbors=30)

# Leiden: resolution is changeable
sc.tl.leiden(ann, resolution=0.2, random_state=random_seed)
print('After clustering\n', ann, '\n')

sc.tl.umap(ann, random_state=random_seed)

# plot leiden
sc.pl.umap(ann, color='leiden', use_raw=False, title='Leiden Results', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_leiden.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by leiden
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='leiden', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_leiden.pdf'))


# ihb_celltypes = {'0':'910-PV+', '1':'Other Ihb neuron', '2':'Other Ihb neuron', '3':'911-Sst+', '4':'Other Ihb neuron', '5':'912-Vip+', '6':'Other Ihb neuron', '7':'Other Ihb neuron', '8':'Other Ihb neuron', '9':'Other Ihb neuron', '10':'Other Ihb neuron', '11':'Other Ihb neuron', '12':'Other Ihb neuron'}
ihb_celltypes = {'0':'910-PV+', '1':'999-Others', '2':'999-Others', '3':'911-Sst+', '4':'999-Others', '5':'912-Vip+', '6':'999-Others', '7':'999-Others', '8':'999-Others', '9':'999-Others', '10':'999-Others', '11':'999-Others', '12':'999-Others'}
ann.obs['celltype'] = (ann.obs['leiden'].map(ihb_celltypes).astype('category'))
# adata.obs['celltype'] = ann.obs.celltype
df2 = ann.obs.celltype

# plot celltype
sc.pl.umap(ann, color='celltype', use_raw=False, title='Cell Type', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_celltype.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by celltype
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='celltype', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_celltype.pdf'))

ann_coor = pd.DataFrame(ann.obsm['X_umap'], columns=['Coor_X', 'Coor_Y'], index=ann.obs['Cell_Index'])
df = pd.concat([ann_coor['Coor_X'], ann_coor['Coor_Y'], ann.obs.Cell_Index, ann.obs.leiden, ann.obs.celltype], axis=1)
df.to_csv(results_dir/(type_factor+'_cellIndex.csv'))
del ann
del ann_coor



## glia
type_factor = 'glia'
cell_cluster = ['1', '5', '7', '9', '11', '12', '13', '15', '16', '17', '18']
print('\n\nPerforming dimension reduction on', type_factor, 'clusters:', cell_cluster, '\n')
ann = adata[adata.obs.leiden.isin(cell_cluster)]

# -- dimension reduction & clustering --
# linear dimension reduction
sc.tl.pca(ann, n_comps=10)
sc.pl.pca_variance_ratio(ann, log=True, show=False)	# to find the best PC number
plt.savefig(results_dir/(type_factor+'_pca_var_ratio.pdf'))

# find neighbors
sc.pp.neighbors(ann, use_rep='X_pca', n_neighbors=50)

# Leiden: resolution is changeable
sc.tl.leiden(ann, resolution=0.2, random_state=random_seed)
print('After clustering\n', ann, '\n')

sc.tl.umap(ann, random_state=random_seed)

# plot leiden
sc.pl.umap(ann, color='leiden', use_raw=False, title='Leiden Results', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_leiden.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by leiden
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='leiden', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_leiden.pdf'))


other_celltypes = {'0':'999-Others', '1':'7-Oligodendrocyte', '2':'999-Others', '3':'6-Enpp2+ neuron', '4':'8-Astrocyte', '5':'8-Astrocyte', '6':'9-Microglia', '7':'999-Others', '8':'7-Oligodendrocyte'}
ann.obs['celltype'] = (ann.obs['leiden'].map(other_celltypes).astype('category'))
# adata.obs['celltype'] = ann.obs.celltype
df3 = ann.obs.celltype

# plot celltype
sc.pl.umap(ann, color='celltype', use_raw=False, title='Cell Type', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_celltype.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by celltype
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(ann, marker_genes, groupby='celltype', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_celltype.pdf'))

ann_coor = pd.DataFrame(ann.obsm['X_umap'], columns=['Coor_X', 'Coor_Y'], index=ann.obs['Cell_Index'])
df = pd.concat([ann_coor['Coor_X'], ann_coor['Coor_Y'], ann.obs.Cell_Index, ann.obs.leiden, ann.obs.celltype], axis=1)
df.to_csv(results_dir/(type_factor+'_cellIndex.csv'))
del ann
del ann_coor 



# =========================
# plot integrated celltype
# =========================
# add adata with celltype
df = pd.concat([df1, df2, df3], axis=0)
df.index = df.index.astype(int)
df = df.sort_index()
df.index = df.index.astype(str)
adata.obs['celltype'] = df

# add adata with three-type celltype
three_celltypes = {'0-L2/3 Ex neuron':'Excitatory neuron', '1-L4 Ex neuron':'Excitatory neuron', '2-L5 Ex neuron':'Excitatory neuron', '3-L6a Ex neuron':'Excitatory neuron', '4-L6b Ex neuron':'Excitatory neuron', '6-Enpp2+ neuron':'Glia cell', '5-Th-Ex':'Excitatory neuron', '910-PV+':'Inhibitory neuron', '911-Sst+':'Inhibitory neuron', '912-Vip+':'Inhibitory neuron', '7-Oligodendrocyte':'Glia cell', '8-Astrocyte':'Glia cell', '9-Microglia':'Glia cell', '999-Others':'Other cell'}
adata.obs['threetype'] = (adata.obs['celltype'].map(three_celltypes).astype('category'))

print('\nIntegrated celltype...\n')
print('Integrated AnnData:\n', adata, '\n')

type_factor = 'integrated'
	
# plot celltype
sc.pl.umap(adata, color='celltype', use_raw=False, title='Cell Type', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_celltype.pdf'), dpi=200, bbox_inches='tight')

# plot gene expression by celltype
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(adata, marker_genes, groupby='celltype', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_celltype.pdf'))

# plot three celltypes
sc.pl.umap(adata, color='threetype', use_raw=False, title='Cell Type', outline_width=(0.3, 0.1), show=False) # palette='tab20'
plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 0), loc=3, frameon=False)
plt.savefig(results_dir/(type_factor+'_umap_threetype.pdf'), dpi=200, bbox_inches='tight')
print('threetype:\n', adata, '\n')

# plot gene expression by three celltype
marker_genes = ['PRISM_1', 'PRISM_2', 'PRISM_3', 'PRISM_4', 'PRISM_5', 'PRISM_6', 'PRISM_7', 'PRISM_8', 'PRISM_9', 'PRISM_10', 'PRISM_11', 'PRISM_12', 'PRISM_13', 'PRISM_14', 'PRISM_15', 'PRISM_16', 'PRISM_17', 'PRISM_18', 'PRISM_19', 'PRISM_20', 'PRISM_21', 'PRISM_22', 'PRISM_23', 'PRISM_24', 'PRISM_25', 'PRISM_26', 'PRISM_27', 'PRISM_28', 'PRISM_29', 'PRISM_30']
sc.pl.dotplot(adata, marker_genes, groupby='threetype', cmap='viridis', use_raw=False, show=False, title='Cell Type Markers')
plt.savefig(results_dir/(type_factor+'_genes_threetype.pdf'))

df = pd.concat([adata_coor['Coor_X'], adata_coor['Coor_Y'], adata.obs.Cell_Index, adata.obs.celltype, adata.obs.leiden, adata.obs.threetype], axis=1)
df.to_csv(results_dir/(type_factor+'_cellIndex.csv'))
'''
