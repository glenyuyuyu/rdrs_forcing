from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import functools
import multiprocessing as mp
import os, sys
from tqdm import *
import pickle
import warnings
warnings.filterwarnings("ignore")


basin_id = '02KB001'

lumped_forcing = True

project_root = Path(os.getcwd())



# Create a folder storing the processed shapefile for subbasins
os.makedirs(Path(project_root / f'shapefiles/{basin_id}'))
os.makedirs(Path(project_root / f'forcing_csvs/{basin_id}'))

for file in os.listdir(project_root / f'routing/{basin_id}'):
    if 'catchment_without_merging_lakes' in file:
        subbasins_shp_path = Path(project_root / f'routing/{basin_id}' / file)
    if 'obs_gauges' in file or 'poi' in file:
        gauge_shp_path = Path(project_root / f'routing/{basin_id}' / file)

subbasins_shp = gpd.read_file(subbasins_shp_path)
subbasins_shp = subbasins_shp.to_crs("EPSG:4326") # set the crs just in case if using OLRP shapefiles
ls_subids = subbasins_shp['SubId'].to_list()

# Check if there are duplicate subbasin ids
if len(ls_subids) != len(set(ls_subids)):
    subbasins_shp = subbasins_shp.dissolve(by='SubId', aggfunc='mean').reset_index()

subbasins_shp = subbasins_shp[['SubId', 'centroid_x', 'centroid_y', 'geometry']]
subbasins_shp['Gauge_ID'] = basin_id
subbasins_shp.to_file(project_root / f'shapefiles/{basin_id}' / f'{basin_id}_subbasins.shp')

forcing_nc_filename = os.listdir(project_root / 'rdrs_downloads')[0]
os.system(f"python derive_grid_weights.py\
                    -i rdrs_downloads/{forcing_nc_filename}\
                    -d 'rlon,rlat' -v 'lon,lat' \
                    -r shapefiles/{basin_id}/{basin_id}_subbasins.shp\
                    -o shapefiles/{basin_id}/{basin_id}_subbasins_grid_weights.txt -a -c 'SubId'")

if lumped_forcing:
    # Define the lumped basin shapefile before removing non-routed subbasin
    basin_shp = subbasins_shp.dissolve(by='Gauge_ID').reset_index()[['Gauge_ID','geometry']]
    gauge_shp = gpd.read_file(gauge_shp_path)
    gauge_shp = gauge_shp.to_crs("EPSG:4326") # set the crs just in case if using OLRP shapefiles
    gauge_shp = gauge_shp.loc[gauge_shp['Obs_NM'] == basin_id].reset_index(drop=True)
    basin_shp['gauge_lat'] = gauge_shp['geometry'].y[0]
    basin_shp['gauge_lon'] = gauge_shp['geometry'].x[0]
    basin_shp['SubId'] = 0
    basin_shp.to_file(project_root / f'shapefiles/{basin_id}' / f'{basin_id}.shp')

    os.system(f"python derive_grid_weights.py\
                    -i rdrs_downloads/{forcing_nc_filename}\
                    -d 'rlon,rlat' -v 'lon,lat' \
                    -r shapefiles/{basin_id}/{basin_id}.shp\
                    -o shapefiles/{basin_id}/{basin_id}_grid_weights.txt -a -c 'SubId'")


# Global variables will be used in the following functions
all_ids = []
all_cells = {}
all_weights = {}

# read the subbasins into the dicts
with open(project_root / f'shapefiles/{basin_id}/{basin_id}_subbasins_grid_weights.txt') as file:
    lines = [line.rstrip() for line in file][7:-1]
for line in lines:
    temp = line.split()
    temp_id = basin_id + '_' + temp[0]
    if temp_id in all_cells:
        all_cells[temp_id].append(int(temp[1]))
        all_weights[temp_id].append(float(temp[2]))
    else:
        all_cells[temp_id] = []
        all_weights[temp_id] = []
        all_cells[temp_id].append(int(temp[1]))
        all_weights[temp_id].append(float(temp[2]))
        all_ids.append(temp_id)


# read the lumped basin into the dicts
if lumped_forcing:
    with open(project_root / f'shapefiles/{basin_id}/{basin_id}_grid_weights.txt') as file:
        lines = [line.rstrip() for line in file][7:-1]

    all_ids.append(basin_id)
    all_cells[basin_id] = []
    all_weights[basin_id] = []

    for line in lines:
        temp = line.split()
        all_cells[basin_id].append(int(temp[1]))
        all_weights[basin_id].append(float(temp[2]))


# Extract the target subbasin id from the above 3 global variables
def cells_weights_collector(subbasin_id : str, cells_dict : dict, weights_dict : dict):
    basin_cells_list = cells_dict[subbasin_id]
    basin_weights_list = weights_dict[subbasin_id]
    return tuple(basin_cells_list), tuple(basin_weights_list)


@functools.lru_cache(maxsize = None)
def load_forcings_to_csv(subbasin_id: str, nc_filename: str, cells: tuple, weights: tuple):
    hourly_nc = xr.open_dataset(project_root / 'rdrs_downloads' / nc_filename)

    one_dim = len(hourly_nc['rlat']) * len(hourly_nc['rlon'])
    variables = list(hourly_nc.keys())
    if 'rotated_pole' in variables:
        variables.remove('rotated_pole')
    temp_ts = hourly_nc['time'].to_series().reset_index(drop=True)
    temp_df = pd.DataFrame(temp_ts)
    temp_df.rename(columns={'time': 'Datetime'}, inplace=True)

    # Define the function to compute the increment from each cell*weight
    def compute_increment(cell_index, all_cells_values, weights):
        increment = all_cells_values[cell_index] * weights[cell_index]
        return increment if not np.isnan(increment) else 0

    # Define a function to compute the variable_value for a given timestep
    def compute_variable_value(timestep, subset, weights):
        all_cells_values = subset[timestep]
        increments = map(lambda cell_index: compute_increment(cell_index, all_cells_values, weights), range(len(cells)))
        return sum(increments)

    # Define a function to process each variable
    def process_variable(v):
        if hourly_nc[v].shape != (len(hourly_nc['time']), len(hourly_nc['rlat']), len(hourly_nc['rlon'])):
            return None
        flat_variable_array = hourly_nc[v].to_numpy().reshape(len(hourly_nc['time']), one_dim)
        subset = np.take(flat_variable_array, cells, 1)
        variable_values = list(map(lambda timestep: compute_variable_value(timestep, subset, weights), range(len(temp_ts))))
        return variable_values
    
    # Apply the process_variable function to each variable and store results in a DataFrame
    results = {v: process_variable(v) for v in variables if process_variable(v) is not None}
    for v, values in results.items():
        temp_df[v] = values
    
    # Save the dataframe to CSV
    subbasin_csv = subbasin_id + '.csv'
    csv_path = Path(project_root / 'forcing_csvs' / f'{basin_id}' / subbasin_csv)
    if csv_path.is_file():     
        basin_df = pd.read_csv(csv_path, index_col=[0])
        basin_df = pd.concat([basin_df, temp_df]).reset_index(drop=True)
        basin_df.to_csv(csv_path)
    else:
        temp_df.to_csv(csv_path)


def clear():
    load_forcings_to_csv.cache_clear()


def wrapper(subbasin_id: str):
    basin_cells, basin_weights = cells_weights_collector(subbasin_id, all_cells, all_weights)

    all_rdrs_files = sorted(os.listdir(project_root / 'rdrs_downloads'))

    for nc_file in all_rdrs_files:
        load_forcings_to_csv(subbasin_id, nc_file, basin_cells, basin_weights)
        clear()


# Set the number of subbasin to be processed in parallel
with mp.Pool(processes = 8) as p:
    max_ = len(all_ids)
    with tqdm(total = max_) as pbar:
        for _ in p.imap_unordered(wrapper, all_ids):
            pbar.update()


