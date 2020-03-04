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
    target_max_BRF = DS.max_BRF.values

    # there is also a placeholder to assign other MAIA water types to -9999.
    print("[surfaceID.py] load MAIA GEOP dataset for the PTA")
    MAIA_PTA_ancillary = xr.open_dataset('../etc/LA_PTA_1KM.nc')
    # MAIA_water_types = MAIA_PTA_ancillary.Landwater_mask.values
    MAIA_lats = MAIA_PTA_ancillary.Latitude
    MAIA_lons = MAIA_PTA_ancillary.Longitude

    # init k-means clustering
    n_clusters = config.getint('kMeans', 'n_cluster')  # including water category, which is always 0
    idx_sza_vza_raz = config.get('kMeans', 'idx_sza_vza_raz').split('/')
    # select data
    print("[surfaceID.py] select smb subsets for clustering")
    sel_data = []
    for idx in idx_sza_vza_raz:
        idx_sza = int(idx.split('.')[0])
        idx_vza = int(idx.split('.')[1])
        idx_raz = int(idx.split('.')[2])
        sel_data.append(target_max_BRF[:, :, idx_sza, idx_vza, idx_raz])

    sel_data = np.array(sel_data)
    input_data = sel_data.reshape(len(idx_sza_vza_raz), -1).transpose()
    print("[surfaceID.py] kMeans input data is in shape of {}".format(input_data.shape))

    # whiten data -- equivalent to using sklearn.preprocessing.StandardScaler
    print("[surfaceID.py] whiten kMeans input data")
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
    print("[surfaceID.py] sort kMeans surface IDs by mean max-BRF")
    grid_mean_BRF = np.nanmean(sel_data, axis=0)

    cluster_mean_BRF = []
    cluster_tot_num = []
    for icluster in np.arange(n_clusters):
        _cluster_BRF = grid_mean_BRF[labels_unsorted == icluster]
        cluster_tot_num.append(len(_cluster_BRF))
        cluster_mean_BRF.append(np.nanmean(_cluster_BRF))
        print(icluster, len(_cluster_BRF), np.nanmean(_cluster_BRF))

    # find the cluster whose cluster_mean_BRF is NaN, change the label to 100 and remove it
    unsorted_clusters = np.dstack([range(n_clusters), cluster_mean_BRF])[0]
    # find it
    nan_cluster = np.arange(n_clusters)[xr.ufuncs.isnan(unsorted_clusters)[:, 1]][0]
    # remove it
    unsorted_clusters_without_nan = np.delete(unsorted_clusters, nan_cluster, axis=0)
    # sort the rest
    sorted_clusters_without_nan = np.array(sorted(unsorted_clusters_without_nan, key=lambda x: x[1]))[:, 0]
    # sort labels (0--n_clusters where category 0 is for unconsidered points)
    labels_sorted = labels_unsorted.copy()
    np.place(labels_sorted, labels_sorted == nan_cluster, 100)
    for i, j in enumerate(sorted_clusters_without_nan):
        np.place(labels_sorted, labels_sorted == j, 101 + i)
    labels_sorted -= 100

    # calculate category-wise BRDF stats (max&min&mean)
    ncats = np.arange(1, n_clusters)
    cat_BRDF_stats = []
    for icat in ncats:
        idx_cat = np.where(labels_sorted == icat)
        cat_BRDF_stats.append(np.max(target_max_BRF[idx_cat], axis=0))
        # cat_BRDF_stats.append(np.min(target_max_BRF[idx_cat], axis=0))
        # cat_BRDF_stats.append(np.mean(target_max_BRF[idx_cat], axis=0))
        # cat_BRDF_stats.append(np.std(target_max_BRF[idx_cat], axis=0))

    # ===========save labels===========
    print("[surfaceID.py] write sorted surface IDs to netcdf")
    sfcID_folder = config.getint('general', 'SfcID_folder')

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
