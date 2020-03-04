"""
Author@Yizhe

Created on Sep 30, 2019

Retrieve the Sample Maximum BRDF (SMB) for all sun-view geometry configurations given the Kiso, Kvol, Kgeo arrays.
Running this script to generate SMB_PTA_xxx.nc data files storing selected maximum BRDFs of each point
within the specified PTA.

Noted on Oct 28, 2019

Save output in hdf5 format will reduce 50% of the output size but cost more time in reading it.
"""

import os
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
def pix_max_BRF(DoY):
    """
    retrieve the maximum BRF for a particular PTA (defined in config.txt file) @ a particular DoY and
    save the results to a netcdf file.

    :param DoY: day of year
    :return:
    """

    # load settings from config.txt file
    print("[pixMaxBRF.py] load settings from config.txt file")
    config = configparser.ConfigParser()
    config.read_file(open('etc/config.txt'))

    cos_sza = [float(i) for i in config.get('sunViewGeometry', 'cos_sza').split(',')]
    vza = [float(i) for i in config.get('sunViewGeometry', 'vza').split(',')]
    raz = [float(i) for i in config.get('sunViewGeometry', 'raz').split(',')]
    sza = np.rad2deg(np.arccos(cos_sza))

    # calculate ROSS-LI kernels
    print("[pixMaxBRF.py] calculate ROSS-LI kernels")
    ROSS_kernels, LI_kernels = kernels_MAIA(sza, vza, raz)

    # open MAIA GEOP dataset for the PTA
    print("[pixMaxBRF.py] open MAIA GEOP dataset for the PTA")
    GEOP_file = config.get('general', 'GEOP_file')
    DS = xr.open_dataset(GEOP_file)
    lat = DS.Latitude  # (1000, 1000)
    lon = DS.Longitude
    lw_mask = DS.Landwater_mask

    # open MAIAC dataset for the PTA at the DOY
    print("[pixMaxBRF.py] open MAIAC dataset for the PTA at the DOY")
    MAIAC_folder = config.get('general', 'MAIAC_folder')
    PTA = config.get('general', 'PTA')
    path_MAIAC = os.path.join(MAIAC_folder, "MAIA_BRDF_{0}_*{1}.hdf".format(PTA, str(DoY).zfill(3)))
    DSS = xr.open_mfdataset(path_MAIAC, concat_dim='yr', combine='nested')

    # choose MODIS band 1 (red)
    k_iso = DSS.Kiso[:, 0, :, :].values  # (15, 1000, 1000)
    k_vol = DSS.Kvol[:, 0, :, :].values
    k_geo = DSS.Kgeo[:, 0, :, :].values

    # do some quality controls (invalid values are set to np.nan so that no target_max_brf will be calculated)
    # check RTLS coefficients
    idx = np.where((k_iso < -1) | (k_vol < -1) | (k_geo < -1))
    k_iso[idx] = np.nan
    k_vol[idx] = np.nan
    k_geo[idx] = np.nan

    # define MAIA water and coastal surface types used in the main loop
    cat_water = [0, 3, 5, 6]
    cat_coastal = [2, 4]

    # main loop
    #                          x   y   SZA VZA RAZ
    # the result array is  (1000, 1000, 10, 15, 12)
    # this shape needs to be changed with changes in {kernels_MAIA} function
    num_y, num_x = lat.shape
    num_SZA = len(sza)
    num_VZA = len(vza)
    num_RAZ = len(raz)

    target_max_brf = np.zeros((num_y, num_x, num_SZA, num_VZA, num_RAZ))
    target_filled = np.ones((num_SZA, num_VZA, num_RAZ)) * -999.
    target_water = np.ones((num_SZA, num_VZA, num_RAZ)) * -998.
    target_coastal = np.ones((num_SZA, num_VZA, num_RAZ)) * -997.

    # line loop
    print("[pixMaxBRF.py] start main loop")
    for i in tqdm(range(num_y)):
        # grid loop
        for j in range(num_x):

            # check MAIA water category mask
            if lw_mask[i, j] in cat_water:
                target_max_brf[i, j] = target_water
            # check MAIA coastal category mask
            elif lw_mask[i, j] in cat_coastal:
                target_max_brf[i, j] = target_coastal
            # calculate max BRF for rest pixels
            else:
                coeff_iso = k_iso[:, i, j]
                coeff_vol = k_vol[:, i, j]
                coeff_geo = k_geo[:, i, j]

                # define number of invalid years by accounting for unavailable coeff_iso
                # (same as coeff_vol and coeff_geo)
                invalid_yrs = np.sum(xr.ufuncs.isnan(k_iso[:, i, j]))
                # do not retrieve BRFs if the number of invalid years is greater than max_invalid_yrs
                max_invalid_yrs = config.getint('mainLoop', 'max_invalid_yrs')
                if invalid_yrs > max_invalid_yrs:
                    # print("sorry to see but there ({}, {}) at DoY-{} is bad...".format(i, j, DoY))
                    target_max_brf[i, j] = target_filled
                else:
                    # SZA loop
                    # for each SZA @ each pixel, retrieve the max BRF for all VZA and RAZ bins
                    for k in range(num_SZA):
                        kern_ross = ROSS_kernels[k]
                        kern_li = LI_kernels[k]
                        target_max_brf[i, j, k] = \
                            max_brf_retrieve_fast(coeff_iso, coeff_vol, coeff_geo, kern_ross, kern_li)

    # save results
    # convert result array to xarray
    print("[pixMaxBRF.py] convert result array to xarray")
    target_max_brf = xr.DataArray(target_max_brf, name='max_BRF',
                                  coords=[('y', np.arange(num_y)),
                                          ('x', np.arange(num_x)),
                                          ('cos_sza', cos_sza), ('vza', vza), ('raz', raz)])
    target_max_brf.encoding['_FillValue'] = -999.

    # save xarray to nc
    print("[pixMaxBRF.py] save xarray to nc")
    SMB_folder = config.get('general', 'SMB_folder')
    smb_file = os.path.join(SMB_folder, 'maxBRF_{}_{}.nc'.format(PTA, str(DoY).zfill(3)))
    target_max_brf.to_netcdf(smb_file, 'w')
    lat.to_netcdf(smb_file, 'a')
    lon.to_netcdf(smb_file, 'a')

    return smb_file
