"""
Author@Yizhe

Created on Sep 30, 2019

Retrieve the Sample Maximum BRDF (SMB) for all sun-view geometry configurations given the Kiso, Kvol, Kgeo arrays.
Running this script to generate SMB_PTA_xxx.nc data files storing selected maximum BRDFs of each point
within the specified PTA.

Noted on Oct 28, 2019

Save output in hdf5 format will reduce 50% of the output size but cost more time in reading it.
"""

import numpy as np
import xarray as xr
import configparser
from tqdm import tqdm
from lib.kernelRTLS import conv_deg_rad, ross_thick, li_sparse


# ----------------Helper functions---------------
def max_brf_retrieve_fast(kisos, kvols, kgeos, kern_ross, kern_li):
    """
    return the maximum BRF (0<= BRF <= 1.6) of a particular location @ a paritular DOY given
    the RTLS parameters.

    :param kisos:
    :param kvols:
    :param kgeos:
    :param kern_ross:
    :param kern_li:
    :return:
    """
    brfs = []
    for kiso, kvol, kgeo in zip(kisos, kvols, kgeos):
        brf = kiso + kvol * kern_ross + kgeo * kern_li
        # simple quality control
        np.place(brf, brf > 1.6, 1.6)
        np.place(brf, brf < 0, 0)
        brfs.append(brf)

    brfs = np.array(brfs)
    max_brf = np.nanmax(brfs, axis=0)

    return max_brf


def kernels_MAIA(SZA, VZA, RAZ):
    """
    return Ross-thick and Li-sparse kernels for the given SZA, VZA, and RAZ arrays.

    :param SZA: Solar zenith angles
    :param VZA: Viewing zenith angles
    :param RAZ: Relative azimuth angles
    :return:
        RT_kernels: Ross-thick kernels in shape of (SZA, VZA, RAZ)
        LS_kernels: Li-sparse kernels in shape of (SZA, VZA, RAZ)
    """
    RT_kernels = np.zeros((len(SZA), len(VZA), len(RAZ)))
    LS_kernels = np.zeros((len(SZA), len(VZA), len(RAZ)))

    for i, isza in enumerate(SZA):
        for j, ivza in enumerate(VZA):
            for k, iraz in enumerate(RAZ):
                rad_sza, rad_vza, rad_phi = conv_deg_rad(isza, ivza, iraz)
                RT_kernels[i, j, k] = ross_thick(rad_sza, rad_vza, rad_phi)
                LS_kernels[i, j, k] = li_sparse(rad_sza, rad_vza, rad_phi)

    return RT_kernels, LS_kernels


# ----------------Main function---------------
def pix_max_BRF(PTA, DoY):
    """
    retrieve the maximum BRF for a particular PTA @ a particular DoY and save the results to
    a netcdf file.

    :param PTA: name of the Primary Target Area
    :param DoY: day of year
    :return:
    """

    # load settings from config.txt file
    config = configparser.ConfigParser()
    config.read_file(open('etc/config.txt'))

    cos_sza = [float(i) for i in config.get('sunViewGeometry', 'cos_sza').split(',')]
    vza = [float(i) for i in config.get('sunViewGeometry', 'vza').split(',')]
    raz = [float(i) for i in config.get('sunViewGeometry', 'raz').split(',')]

    print(cos_sza)
    # # Get RTLS kernels, once for all
    # # change sun-view configurations here when necessary
    # # --------
    # # Current settings:
    # # cos(SZA): 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95
    # # correspond to SZA: 87.1, 81.4, 75.5, 69.5, 63.3, 56.6, 49.5, 41.4, 31.8, 18.2 (degree)
    # # VZA: 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5 (degree)
    # # RAZ: 7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5, 112.5, 127.5, 142.5, 157.5, 172.5 (degree)
    # cos_sza = 0.05 + np.arange(10.) / 10.
    # SZA = np.rad2deg(np.arccos(cos_sza))
    # VZA = np.arange(0, 73, 5) + 2.5
    # RAZ = np.arange(0, 180, 15) + 7.5
    # ROSS_kernels, LI_kernels = kernels_MAIA(SZA, VZA, RAZ)
    #
    # # Open MAIA GEOP dataset for the PTA
    # path_GEOP = "MAIA/{}_PTA_1KM.nc".format(PTA)
    # DS = xr.open_dataset(path_GEOP)
    # lat = DS.Latitude  # (1000, 1000)
    # lon = DS.Longitude
    # lw_mask = DS.Landwater_mask
    #
    # # Open MAIAC dataset for the PTA at the DOY
    # path_MAIAC = "MAIAC/{0}/MAIA_BRDF_{0}_*{1}.hdf".format(PTA, str(DoY).zfill(3))
    # DSS = xr.open_mfdataset(path_MAIAC, concat_dim='yr')
    #
    # k_iso = DSS.Kiso[:, 0, :, :].values  # (15, 1000, 1000)
    # k_vol = DSS.Kvol[:, 0, :, :].values
    # k_geo = DSS.Kgeo[:, 0, :, :].values
    #
    # # ============quality control============
    # # all invalid values are set to np.nan so that no target_max_brf will be calculated
    #
    # # Check RTLS coefficients
    # idx = np.where((k_iso < -1) | (k_vol < -1) | (k_geo < -1))
    # k_iso[idx] = np.nan
    # k_vol[idx] = np.nan
    # k_geo[idx] = np.nan
    #
    # # Define MAIA water and coastal surface types
    # # used in the main loop
    # cat_water = [0, 3, 5, 6]
    # cat_coastal = [2, 4]
    #
    # # ===========main process===========
    # #                          x   y   SZA VZA RAZ
    # # the result array is  (1000, 1000, 10, 15, 12)
    # # this shape needs to be changed with changes in {kernels_MAIA} function
    # num_y, num_x = lat.shape
    # num_SZA = len(SZA)
    # num_VZA = len(VZA)
    # num_RAZ = len(RAZ)
    #
    # target_max_brf = np.zeros((num_y, num_x, num_SZA, num_VZA, num_RAZ))
    # target_filled = np.ones((num_SZA, num_VZA, num_RAZ)) * -999.
    # target_water = np.ones((num_SZA, num_VZA, num_RAZ)) * -998.
    # target_coastal = np.ones((num_SZA, num_VZA, num_RAZ)) * -997.
    #
    # # line loop
    # for i in tqdm(range(num_y)):
    #     # grid loop
    #     for j in range(num_x):
    #
    #         # check MAIA water category mask
    #         if lw_mask[i, j] in cat_water:
    #             target_max_brf[i, j] = target_water
    #         # check MAIA coastal category mask
    #         elif lw_mask[i, j] in cat_coastal:
    #             target_max_brf[i, j] = target_coastal
    #         # calculate max BRF for rest pixels
    #         else:
    #             coeff_iso = k_iso[:, i, j]
    #             coeff_vol = k_vol[:, i, j]
    #             coeff_geo = k_geo[:, i, j]
    #
    #             # define number of invalid years by accounting for unavailable coeff_iso
    #             # (same as coeff_vol and coeff_geo)
    #             invalid_yrs = np.sum(xr.ufuncs.isnan(k_iso[:, i, j]))
    #             # do not retrieve BRFs if the number of invalid years is greater than 5
    #             if invalid_yrs > 5:
    #                 # print("sorry to see but there ({}, {}) at DoY-{} is bad...".format(i, j, DoY))
    #                 target_max_brf[i, j] = target_filled
    #             else:
    #                 # SZA loop
    #                 # for each SZA @ each pixel, retrieve the max BRF for all VZA and RAZ bins
    #                 for k in range(num_SZA):
    #                     kern_ross = ROSS_kernels[k]
    #                     kern_li = LI_kernels[k]
    #                     target_max_brf[i, j, k] = max_brf_retrieve_fast(coeff_iso, coeff_vol, coeff_geo, kern_ross,
    #                                                                     kern_li)
    #
    # # ===========save results===========
    # # convert result array to xarray
    # target_max_brf = xr.DataArray(target_max_brf, name='max_BRF',
    #                               coords=[('y', np.arange(num_y)), ('x', np.arange(num_x)), ('cos_sza', cos_sza),
    #                                       ('vza', VZA), ('raz', RAZ)])
    # target_max_brf.encoding['_FillValue'] = -999.
    #
    # # save xarray to nc
    # out_nc = 'MAIAC_SMB/maxBRF_{}_{}.nc'.format(PTA, str(DoY).zfill(3))
    # target_max_brf.to_netcdf(out_nc, 'w')
    # lat.to_netcdf(out_nc, 'a')
    # lon.to_netcdf(out_nc, 'a')
    #
    # return out_nc


if __name__ == '__main__':

    for doy in np.arange(16, 366, 8):
        pix_max_BRF('LA', doy)
