import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree

BASE_DIR = Path(r'F:\spatial_data\processed')
RUN_ID = '20231226_FFPE_trial5_PRISM_3um_NaBH4_LED_after_FISH'
src_dir = BASE_DIR / f'{RUN_ID}_processed'
stc_dir = src_dir / 'stitched'
read_dir = src_dir / 'readout'
seg_dir = src_dir / 'segmented'
visual_dir = src_dir / 'visualization'
visual_dir.mkdir(exist_ok=True)

# Read nucleus position
centroids = pd.read_csv(seg_dir/'centroids_all.csv', header=None).to_numpy()

# Assign RNA to its nearest nucleus
rna_df = pd.read_csv(read_dir/'mapped_genes.csv')
rna_pos = rna_df[['Y', 'X']].to_numpy()
tree = KDTree(centroids)
distances, indices = tree.query(rna_pos, k=1, distance_upper_bound=100)
rna_df['Cell Index'] = indices
rna_df = rna_df[rna_df['Cell Index'] < centroids.shape[0]]

# Generate expression matrix
match_df = rna_df.copy()
match_df['Count'] = np.ones(len(match_df))
match_df_group = match_df[['Cell Index','Gene','Count']].groupby(['Cell Index','Gene']).count()
matrix = match_df_group.unstack().fillna(0)
matrix.columns = matrix.columns.droplevel()
matrix.columns.name = None
matrix.index.name = None

matrix.to_csv(seg_dir / 'expression_matrix.csv')