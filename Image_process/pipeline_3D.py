import os
import glob
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')
from tqdm import tqdm
# from unittest.mock import patch

from lib.utils.io_utils import get_tif_list
from lib.fstack import stack_cyc
from lib.cidre import cidre_correct, cidre_walk
from lib.register import register_meta
from lib.stitch import patch_tiles
from lib.stitch import template_stitch
from lib.stitch import stitch_offset

from lib.register import register_manual
from lib.stitch import stitch_manual
from lib.os_snippets import try_mkdir
    
from skimage.io import imread
from skimage.io import imsave
from skimage.util import img_as_uint
from skimage.transform import resize


def resize_pad(img, size):
    img_resized = resize(img, size, anti_aliasing=True)
    img_padded = np.zeros(img.shape)
    y_start, x_start = (img.shape[0] - size[0]) // 2, (img.shape[1] - size[1]) // 2
    img_padded[y_start:y_start+size[0], x_start:x_start+size[1]] = img_resized
    img_padded = img_as_uint(img_padded)
    return img_padded


def resize_dir(in_dir, out_dir, chn):
    Path(out_dir).mkdir(exist_ok=True)
    chn_sizes = {'cy3': 2302, 'TxRed': 2303, 'FAM': 2301, 'DAPI': 2300}
    size = chn_sizes[chn]
    im_list = list(Path(in_dir).glob(f'*.tif'))
    for im_path in tqdm(im_list, desc=Path(in_dir).name):
        im = imread(im_path)
        im = resize_pad(im, (size, size))
        imsave(Path(out_dir)/im_path.name, im, check_contrast=False)


def resize_batch(in_dir, out_dir):
    try_mkdir(out_dir)
    cyc_paths = list(Path(in_dir).glob('cyc_*_*'))
    for cyc_path in cyc_paths:
        chn = cyc_path.name.split('_')[-1]
        if chn == 'cy5':
            shutil.copytree(cyc_path, Path(out_dir)/cyc_path.name)
        else:
            resize_dir(cyc_path, Path(out_dir)/cyc_path.name, chn)


SRC_DIR = Path(r'/mnt/data/raw_images')
BASE_DIR = Path(r'/mnt/data/local_processed_data')
RUN_ID = '20240308_FFPE_NABH_7_UV_NaBH4'
src_dir = SRC_DIR / RUN_ID
dest_dir = BASE_DIR / f'{RUN_ID}_processed'

# 2D workflow
aif_dir = dest_dir / 'focal_stacked'
sdc_dir = dest_dir / 'background_corrected'
rgs_dir = dest_dir / 'registered'
stc_dir = dest_dir / 'stitched'
rsz_dir = dest_dir / 'resized'
read_dir = dest_dir / 'readout'

# 3D workflow
cid_dir = dest_dir / 'cidre'
air_dir = dest_dir / 'airlocalize_stack'

# process 2d
def process_2d():
    # raw_cyc_list = list(src_dir.glob('cyc_*'))
    # for cyc in raw_cyc_list:
    #   cyc_num = int(cyc.name.split('_')[1])
    #   stack_cyc(src_dir, aif_dir, cyc_num)

    cidre_walk(str(aif_dir), str(sdc_dir))

    rgs_dir.mkdir(exist_ok=True)
    ref_cyc = 1
    ref_chn = 'cy3'
    ref_chn_1 = 'cy5'
    ref_dir = sdc_dir / f'cyc_{ref_cyc}_{ref_chn}'
    im_names = get_tif_list(ref_dir)

    meta_df = register_meta(str(sdc_dir), str(rgs_dir), ['cy3', 'cy5', 'DAPI'], im_names, ref_cyc, ref_chn)
    meta_df.to_csv(rgs_dir / 'integer_offsets.csv')
    register_manual(rgs_dir/'cyc_10_DAPI', sdc_dir/'cyc_11_DAPI', rgs_dir/'cyc_11_DAPI') ##

    # register_manual(rgs_dir / 'cyc_1_cy3', sdc_dir / 'cyc_1_cy5', rgs_dir / 'cyc_1_cy5') #
    # register_manual(rgs_dir / 'cyc_1_cy3', sdc_dir / 'cyc_1_FAM', rgs_dir / 'cyc_1_FAM')
    # register_manual(rgs_dir / 'cyc_1_cy3', sdc_dir / 'cyc_1_TxRed', rgs_dir / 'cyc_1_TxRed')
    # register_manual(rgs_dir / 'cyc_1_cy3', sdc_dir / 'cyc_1_DAPI', rgs_dir / 'cyc_1_DAPI')  # 0103 revised! Please remove this !
    
    patch_tiles(rgs_dir/f'cyc_{ref_cyc}_{ref_chn}', 6 * 7)
    # resize_batch(rgs_dir, rsz_dir)
    stc_dir.mkdir(exist_ok=True)
    template_stitch(rgs_dir/f'cyc_{ref_cyc}_{ref_chn_1}', stc_dir, 6, 7)
    
    # offset_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)
    # stitch_offset(rgs_dir, stc_dir, offset_df)

# process 3d
# Define your per-slice and per-stack programs
def process_slice(slice_2d, channel): 
    if channel != 'cy5':
        # resize and pad the slice
        chn_sizes = {'cy3': 2302, 'txred': 2303, 'fam': 2301, 'dapi': 2300}
        size = chn_sizes[channel]
        slice_2d = resize_pad(slice_2d, (size, size))
    return slice_2d  # Placeholder


# Adjust shift_correction
# def shift_correction(signal_df, shift_df, tile, cyc, ref_cyc=1):
#     adjusted_signals = []
#     file = f'FocalStack_{tile:03d}.tif'
#     for _, signal_row in signal_df.iterrows():    
#         local_x, local_y = signal_row['x_in_pix'], signal_row['y_in_pix']

#         # Apply shift if not reference cycle
#         if cyc != ref_cyc:
#             shift_entry = shift_df.loc[cyc, file]  # Assuming shift_df is indexed by cycle and file
#             y_shift, x_shift = map(int, shift_entry.split(' '))
#             current_x = local_x + x_shift
#             current_y = local_y + y_shift
#         else: current_x, current_y = local_x, local_y

#         adjusted_signals.append((current_y, current_x))

#     xy_adjusted = pd.DataFrame(adjusted_signals, columns=['y_in_pix', 'x_in_pix'])
#     signal_df.loc[:, ['y_in_pix', 'x_in_pix']] = xy_adjusted[['y_in_pix', 'x_in_pix']].values    
#     return signal_df


def shift_correction(signal_df, shift_df, tile, cyc, ref_cyc=1):
    file = f'FocalStack_{tile:03d}.tif'
    signal_df = signal_df.copy()
    # Handle the case where the index might not exist
    try:
        if cyc != ref_cyc:
            shift_entry = shift_df.loc[(cyc, file)]
            y_shift, x_shift = map(int, shift_entry.split(' '))
            signal_df.loc[:, 'x_in_pix'] += x_shift
            signal_df.loc[:, 'y_in_pix'] += y_shift
    except KeyError:
        print(f"Cycle {cyc} or file {file} not found in shift_df.")
        # Optionally, handle the absence of the key more gracefully here
        pass
    
    return signal_df


# def stitch_3d(signal_df, meta_df, tile):
#     adjusted_signals = []
#     file = f'FocalStack_{tile:03d}.tif'
#     # Find the metadata row for this file to get its global position
#     for _, signal_row in signal_df.iterrows(): 
#         meta_row = meta_df.loc[meta_df['file'] == file].iloc[0]
#         local_x, local_y = signal_row['x_in_pix'], signal_row['y_in_pix']
#         global_x_start, global_y_start = meta_row['x'], meta_row['y']
#         global_x = global_x_start + local_x
#         global_y = global_y_start + local_y
#         adjusted_signals.append((global_y, global_x))

#     xy_adjusted = pd.DataFrame(adjusted_signals, columns=['y_in_pix', 'x_in_pix'])
#     signal_df.loc[:, ['y_in_pix', 'x_in_pix']] = xy_adjusted[['y_in_pix', 'x_in_pix']].values    
#     return signal_df


def stitch_3d(signal_df, meta_df, tile):
    file = f'FocalStack_{tile:03d}.tif'
    # Get the metadata row for this file to get its global position
    meta_row = meta_df.loc[file]
    # Get global positions
    global_x_start, global_y_start = meta_row['x'], meta_row['y']
    # Vectorized calculation to update signal_df directly
    signal_df['y_in_pix'] = signal_df['y_in_pix'] + global_y_start
    signal_df['x_in_pix'] = signal_df['x_in_pix'] + global_x_start
    
    return signal_df


import sys
import re
from collections import defaultdict

import tifffile
from lib.stitch import read_meta
from lib.AIRLOCALIZE.airlocalize import airlocalize
from skimage.morphology import white_tophat
import numpy as np

def create_ellipsoid_kernel(x_radius, y_radius, z_radius):
    x = np.arange(-x_radius, x_radius+1)
    y = np.arange(-y_radius, y_radius+1)
    z = np.arange(-z_radius, z_radius+1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    ellipsoid = (xx**2 / x_radius**2) + (yy**2 / y_radius**2) + (zz**2 / z_radius**2)

    kernel = ellipsoid <= 1
    return kernel.astype(np.uint8)


extract_points_cycle = ['C001', 'C002', 'C003', 'C004']
CHANNELS = ['cy3', 'cy5']
TOPHAT_STRUCTURE = create_ellipsoid_kernel(2, 3, 3)


def process_3d():
    # generate corrected 3d image of each tile
    # cidre_correct(str(src_dir), str(cid_dir))

    stack_name = dict()
    file_groups = defaultdict(list)
    for cyc_folder in glob.glob(os.path.join(src_dir, 'cy*')):
        for file_path in glob.glob(os.path.join(cyc_folder, '*.tif')):
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            cycle, tile, channel = parts[0], parts[1], parts[2]
            if channel in CHANNELS:
                z_index = int(filename.split('Z')[-1].split('.')[0])
                file_groups[(cycle, tile, channel)].append((z_index, file_path))
                if tile in stack_name: stack_name[tile].add(cycle)
                else: stack_name[tile] = set()
    
    stack_name = {key: sorted(list(value), key=lambda x: int(x[1:])) for key, value in stack_name.items()}
    file_groups = {k: sorted(v) for k, v in file_groups.items()}  # Sort by Z index within each group


    # # create 3d file from slices
    # for (cycle, tile, channel), files in tqdm(file_groups.items(), desc='Processing stacks'):
    #     stack = np.array([process_slice(imread(file_path), channel) for _, file_path in files])
    #     os.makedirs(air_dir / tile / cycle, exist_ok=True)
    #     tifffile.imwrite(air_dir / tile / cycle / f"{channel.lower()}.tif", stack)

    # extract spot candidates from cyc1-4
    # for tile in tqdm(stack_name.keys(), desc='Detecting candidate points', position=0, leave=True):
    #     df = pd.DataFrame()
    #     df.to_csv(air_dir/tile/'combined_candidates.csv', index=False)
    #     for cycle in tqdm(extract_points_cycle, desc=f'Processing cycles for tile {tile}', position=1, leave=False):
    #         tile_cycle_dir = air_dir  / tile / cycle
    #         # perform airlocalization
    #         airlocalize(
    #             parameters_filename='/mnt/data/processing_codes/SPRINT_analysis/lib/AIRLOCALIZE/parameters.yaml', 
    #             default_parameters='/mnt/data/processing_codes/SPRINT_analysis/lib/AIRLOCALIZE/parameters_default.yaml',
    #             update={'dataFileName': tile_cycle_dir, 'saveDirName': tile_cycle_dir, 'verbose':False})
            
    #         spots_file = [_ for _ in os.listdir(tile_cycle_dir) if _.endswith('spots.csv')]
    #         df = pd.read_csv(air_dir / tile / 'combined_candidates.csv')
            
    #         if len(df) > 0: df = pd.concat([df] + [pd.read_csv(tile_cycle_dir / file) for file in spots_file], axis=0)
    #         else: df = pd.concat([pd.read_csv(tile_cycle_dir / file) for file in spots_file], axis=0)
    #         df.to_csv(air_dir/tile/'combined_candidates.csv', index=False)


    # multi-channel read
    shift_df = pd.read_csv(rgs_dir / 'integer_offsets.csv', index_col=0)

    for tile in tqdm(stack_name.keys(), desc='Reading spots', position=0, leave=True):
        combined_candidates = pd.read_csv(air_dir / tile / 'combined_candidates.csv')
        intensity_read = combined_candidates[['z_in_pix', 'y_in_pix', 'x_in_pix']].round().astype(np.uint16).drop_duplicates()
        intensity_read = intensity_read.reset_index()
        for cycle in tqdm(stack_name[tile], desc=f'Processing cycles for tile {tile}', position=1, leave=False):
            with tifffile.TiffFile(air_dir / tile / cycle / 'cy3.tif') as tif:
                shape = tif.series[0].shape

            coordinates = intensity_read[['z_in_pix', 'y_in_pix', 'x_in_pix']]
            coordinates = shift_correction(coordinates, shift_df, tile=int(tile[1:]), cyc=int(cycle[1:]), ref_cyc=1)
            coordinates = coordinates[
                (0 <= coordinates['z_in_pix'] ) & (coordinates['z_in_pix'] < shape[0]) &
                (0 <= coordinates['y_in_pix'] ) & (coordinates['y_in_pix'] < shape[1]) &
                (0 <= coordinates['x_in_pix'] ) & (coordinates['x_in_pix'] < shape[2]) ]
            z_coords = coordinates['z_in_pix'].to_numpy()
            y_coords = coordinates['y_in_pix'].to_numpy()
            x_coords = coordinates['x_in_pix'].to_numpy()

            for channel in CHANNELS:
                image = white_tophat(imread(air_dir / tile / cycle / f'{channel}.tif'), selem=TOPHAT_STRUCTURE)
                coordinates[f'{cycle}_{channel}'] = image[z_coords, y_coords, x_coords]
                intensity_read['cyc_{}_{}'.format(int(cycle[1:]), channel)] = coordinates[f'{cycle}_{channel}']
        intensity_read = intensity_read.dropna()
        intensity_read.to_csv(air_dir / tile / 'intensity_local.csv', index=None)
    

    # stitch the intensity
    meta_df = read_meta(stc_dir)
    pattern = r'\((\d+)\, *(\d+)\)'
    meta_df['match'] = meta_df['position'].apply(lambda x: re.match(pattern, x))
    meta_df['y'] = meta_df['match'].apply(lambda x: int(x.group(2)))
    meta_df['x'] = meta_df['match'].apply(lambda x: int(x.group(1)))
    meta_df.set_index('file', inplace=True)
    intensity = None
    for tile in tqdm(stack_name.keys(), desc='Stitching'):
        signal_df = pd.read_csv(air_dir / tile / 'intensity_local.csv')
        if intensity is None: intensity = stitch_3d(signal_df, meta_df, tile=int(tile[1:]))
        else: intensity = pd.concat([intensity, stitch_3d(signal_df, meta_df, tile=int(tile[1:]))])
    intensity.index = pd.RangeIndex(start=0, stop=len(intensity), step=1)
    intensity.to_csv(read_dir / 'intensity_all.csv', index=None)


def main():
    # process_2d()
    process_3d()


if __name__ == '__main__':
    main()