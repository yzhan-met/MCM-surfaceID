"""
Author@Yizhe

Created on Oct 1, 2019

Retrieve the surface ID from the max. BRF dataset.
Running this script to generate SurfaceID_PTA_xxx.nc data files storing the surface ID and the representative BRDF
for each surface category for the specified PTA.

"""

import os
import numpy as np
import xarray as xr
import configparser
from sklearn.cluster import MiniBatchKMeans


def surface_id(file_smb):
    """
    generate surface ID for the given max_BRF dataset.

    :param file_smb:
    :return:
    """
    config = configparser.ConfigParser()
    config.read_file(open('etc/config.txt'))

    # use xarray.open_mfdataset to implement dask multi-processor
    # the result array -- target_max_brf contains NaN
    # remember to change NaN values to -9999. before running K-means clustering
    print("[surfaceID.py] load intermediate smb file for the PTA")
    DS = xr.open_mfdataset(file_smb)

    # there is also a placeholder to assign other MAIA water types to -9999.
    # print("[surfaceID.py] load MAIA GEOP dataset for the PTA")
    # MAIA_PTA_ancillary = xr.open_dataset('../etc/LA_PTA_1KM.nc')
    # MAIA_water_types = MAIA_PTA_ancillary.Landwater_mask.values
    # MAIA_lats = MAIA_PTA_ancillary.Latitude
    # MAIA_lons = MAIA_PTA_ancillary.Longitude

    # init k-means clustering
    n_clusters = config.getint('kMeans', 'n_cluster')  # including water category, which is always 0
    sel_cos_sza = [float(i) for i in config.get('kMeans', 'sel_cos_sza').split(',')]
    sel_vza = [float(i) for i in config.get('kMeans', 'sel_vza').split(',')]
    sel_raz = [float(i) for i in config.get('kMeans', 'sel_raz').split(',')]

    # select data
    print("[surfaceID.py] select smb subsets for clustering")
    sel_data = DS.max_BRF.sel(vza=sel_vza,
                              raz=sel_raz)
    filled_sel_data = sel_data.fillna(-996.)

    sample_data = filled_sel_data[:, :, 0, 0, 0]
    idx_coastal = np.where(sample_data == -997.)
    idx_water = np.where(sample_data == -998.)
    idx_unknown = np.where(sample_data == -996.)

    stacked_filled_sel_data = sel_data.stack(i=('y', 'x'), z=('cos_sza', 'vza', 'raz'))
    input_data = stacked_filled_sel_data.values

    print("[surfaceID.py] kMeans input data is in shape of {}".format(input_data.shape))

    # whiten data -- equivalent to using sklearn.preprocessing.StandardScaler
    print("[surfaceID.py] whiten kMeans input data")
    np.place(input_data, input_data < -900., np.nan)
    input_data_mean = np.nanmean(input_data, axis=0)
    input_data_std = np.nanstd(input_data, axis=0)
    whitened_data = (input_data - input_data_mean) / input_data_std
    idx_nan = xr.ufuncs.isnan(whitened_data)
    whitened_data[idx_nan] = -9999.

    print("[surfaceID.py] run kMeans model")
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=0,
        batch_size=config.getint('kMeans', 'batch_size'),
        verbose=0,
        max_no_improvement=config.getint('kMeans', 'max_no_improvement')).fit(whitened_data)

    labels_unsorted = minibatch_kmeans.labels_.reshape(1000, 1000)

    # sort labels
    print("[surfaceID.py] sort kMeans surface IDs by grid mean BRF")

    # calculate grid_mean_BRF
    grid_mean_BRF = stacked_filled_sel_data.unstack('i').mean(axis=0).values

    # use grid_mean_BRF to get cluster_mean_BRF_unsorted
    cluster_mean_BRF_unsorted = []
    for icluster in np.arange(n_clusters):
        _cluster_BRF = grid_mean_BRF[labels_unsorted == icluster]
        cluster_mean_BRF_unsorted.append(np.nanmean(_cluster_BRF))

    # get labels_sorted based on cluster_mean_BRF_unsorted (small to large)
    # assign a number to each cluster_mean_BRF
    clusters_unsorted = np.dstack([range(n_clusters), cluster_mean_BRF_unsorted])[0]
    # sort the rest
    clusters_sorted = np.array(sorted(clusters_unsorted, key=lambda x: x[1]))[:, 0]
    # sort labels (0--n_clusters where category 0 is for unconsidered points)
    labels_sorted = labels_unsorted.copy()
    for i, j in enumerate(clusters_sorted):
        np.place(labels_sorted, labels_unsorted == j, 101 + i)
    labels_sorted -= 100
    # assign fill-in labels
    labels_sorted[idx_water] = 0
    labels_sorted[idx_coastal] = 1
    labels_sorted[idx_unknown] = 1

    sorted_cluster_std_BRF = []
    sorted_cluster_mean_BRF = []
    sorted_cluster_tot_num = []

    # plus 1 because adding water type (0)
    for icluster in np.arange(n_clusters + 1):
        _cluster_BRF = grid_mean_BRF[labels_sorted == icluster]

        sorted_cluster_mean_BRF.append(np.nanmean(_cluster_BRF))
        sorted_cluster_std_BRF.append(np.nanstd(_cluster_BRF))
        sorted_cluster_tot_num.append(len(_cluster_BRF))
        print("{:2d} {:6d} {:.2f} {:.2f}".format(icluster, sorted_cluster_tot_num[icluster],
                                           sorted_cluster_mean_BRF[icluster], sorted_cluster_std_BRF[icluster]))

    # ===========save labels===========
    print("[surfaceID.py] write sorted surface IDs to netcdf")
    MAIA_lats = DS.Latitude
    MAIA_lons = DS.Longitude
    sfcID_folder = config.get('general', 'SfcID_folder')

    sfcID_file = os.path.join(sfcID_folder, file_smb.replace("maxBRF", "surfaceID"))
    MAIA_lats.to_netcdf(sfcID_file, 'w')
    MAIA_lons.to_netcdf(sfcID_file, 'a')
    MAIA_labels = xr.DataArray(labels_sorted, name='surface_ID',
                               coords=[('y', np.arange(1000)), ('x', np.arange(1000))])
    MAIA_labels.to_netcdf(sfcID_file, 'a')

    # ===========save BRDFs===========
    # MAIA_BRDFs = xr.DataArray(cat_BRDF_stats, name='category_BRDF',
    #                           coords=[('sfc_id', ncats), ('sza', DS.sza.values),
    #                                   ('vza', DS.vza.values), ('raz', DS.raz.values)])
    # MAIA_BRDFs.to_netcdf(sfcID_file, 'a')

    return sfcID_file
