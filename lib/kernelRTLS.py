"""
Author@Yizhe

Created on Oct 1, 2019

The Ross-Li BRDF model is used here, which in principle has three components:
    1. isotropic scattering
    2. volumetric scattering as from horizontally homogeneous leaf canopies
    3. geometric-optical surface scattering as from scenes containing 3-D objects that 
        cast shadows and are mutually obscurred from view at off-nadir angles

One may think of the volumetric scattering term as expressing effects caused by the small
(interleaf) gaps in a canopy whereas the geometric-optical term express effects caused
by the larger (intercrown) gaps.
"""

import numpy as np


def brf_forward(geoms, params):
    """Calculate surface BR(D?)F.
    
    The calculation is based on ROSS-LI surface BRDF kernels.
    
    Arguments:
        geoms {[float32, float32, float32]} -- (in degree) Solar zenith angle, Viewing zenith angle, Relative azimuth angle
        params {[float32, float32, float32]} -- k_iso, k_vol, k_geo
    
    Returns:
        [float32] -- BR(D?)F
    """
    rad_sza, rad_vza, rad_phi = conv_deg_rad(geoms[0], geoms[1], geoms[2])

    r_thick = ross_thick(rad_sza, rad_vza, rad_phi)

    li_sp = li_sparse(rad_sza, rad_vza, rad_phi)

    refl = params[0] + params[1] * r_thick + params[2] * li_sp
    # print(refl, r_thick, li_sp)
    return refl


# -----------------------------------------------------------------------------------

def conv_deg_rad(sza_deg, vza_deg, phi_deg):
    """
    convert sun-view geometry from degree to radians

    :param sza_deg: solar zenith angle array/value
    :param vza_deg: viewing zenith angle array/value
    :param phi_deg: relative azimuth angle array/value
    :return:
    """
    sza_rad = np.deg2rad(sza_deg)
    vza_rad = np.deg2rad(vza_deg)
    phi_rad = np.deg2rad(phi_deg)
    return sza_rad, vza_rad, phi_rad


def cal_phaang(sza_rad, vza_rad, phi_rad):
    """
    calculate scattering angle
    
    The following formula was applied here:
        cos_pha = cos_vza*cos_sza + sin_vza*sin_sza*cos_phi
    
    Note that sza, vza, and phi are in radians

    Arguments:
        sza_rad {[type]} -- [description]
        vza_rad {[type]} -- [description]
        phi_rad {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    cos_pha = np.cos(sza_rad) * np.cos(vza_rad) + np.sin(sza_rad) * np.sin(vza_rad) * np.cos(phi_rad)
    pha_rad = np.arccos(cos_pha)
    return pha_rad


def ross_thick(sza_rad, vza_rad, phi_rad):
    """
    A suitable K_vol was derived by Roujean et al. (1992), called the RossThick kernel for its
    assumption of a dense leaf canopy. It is a single-scattering approx. of RT by Ross (1981) 
    consisting of a layer of small scatterers with uniform leaf angle distribution, a Lambertian
    background, and equal leaf transmittance and reflectance.

    Its form, normalized to 0 for SZA=0, VZA=0 is used here.

    Arguments:
        sza_rad {[type]} -- [description]
        vza_rad {[type]} -- [description]
        phi_rad {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    pha_rad = cal_phaang(sza_rad, vza_rad, phi_rad)

    ross_kernel = ((np.pi / 2 - pha_rad) * np.cos(pha_rad) + np.sin(pha_rad)) / \
                  (np.cos(sza_rad) + np.cos(vza_rad)) - np.pi / 4
    # (1+1/(1+pha_rad/np.deg2rad(1.5)))
    return ross_kernel


def li_sparse(sza_rad, vza_rad, phi_rad):
    """
    A suitable K_geo was derived by Wanner et al. (1995), called the LiSparse kernel for its 
    assumption of a sparse ensemble of surface objects casting shadows on the background, which 
    is also assumed as Lambertian. The kernel is derived from the geomtric-optical mutual shadowing
    BRDF model by Li and Strahler (1992). It is given by the proportions of sunlit and shaded scene 
    components in a scene consisting of randomly located spheroids of height-to-center-of-crown {h} 
    and crown vertical to horizontal radius ratio {b/r}.

    The original form of this kernel is not reciprocal in SZA and VZA, a property that is expected
    from homogeneous natural surfaces viewed at coarse spatial scale. The main reason for this 
    nonreciprocity is that the scene component reflectances are assumed to be constants independent
    of SZA. If the sunlit component is simply assumed to vary as 1/cos_SZA, the kernel takes on 
    the reciprocal form given here, to be called LiSparse-R.

    Arguments:
        sza_rad {[type]} -- [description]
        vza_rad {[type]} -- [description]
        phi_rad {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    # const
    b_r = 1.0
    h_b = 2.0

    def ang_prime(angle):
        # calculate prime vza and sza
        tan_prime = b_r * np.tan(angle)
        try:
            np.place(tan_prime, tan_prime < 0, 0.)
        except:
            tan_prime = np.max([0, tan_prime])
        rad_prime = np.arctan(tan_prime)
        return rad_prime

    def cal_distance(rad_sza, rad_vza, rad_phi):
        # calculate the D distance
        tmp_d = np.tan(rad_sza) ** 2 + np.tan(rad_vza) ** 2 - 2 * np.tan(rad_sza) * np.tan(rad_vza) * np.cos(rad_phi)
        try:
            np.place(tmp_d, tmp_d < 0, 0.)
        except:
            tmp_d = np.max([0, tmp_d])
        distance = np.sqrt(tmp_d)
        return distance

    # overlap, which is the overlap area between the view and solar shadows, is calculated based on rad_t, pri_sza,
    # and pri_vza.
    pri_sza = ang_prime(sza_rad)
    pri_vza = ang_prime(vza_rad)
    sec_pri_vza = 1 / np.cos(pri_vza)
    sec_pri_sza = 1 / np.cos(pri_sza)

    #   to calculate radiant t, we need to firstly calculate distance (D).
    D = cal_distance(pri_sza, pri_vza, phi_rad)
    # cos_t should be constrained to the range [-1, 1], as values outside of this range imply no overlap and should
    # be disregarded.
    cos_t = h_b * np.sqrt(D ** 2 + (np.tan(pri_sza) * np.tan(pri_vza) * np.sin(phi_rad)) ** 2) / (
            sec_pri_vza + sec_pri_sza)
    try:
        np.place(cos_t, cos_t > 1., 1.)
        np.place(cos_t, cos_t < -1., -1.)
    except:
        cos_t = np.max([-1, np.min([1, cos_t])])
    rad_t = np.arccos(cos_t)
    #   calculate the overlap distance
    overlap = (1 / np.pi) * (rad_t - np.sin(rad_t) *
                             np.cos(rad_t)) * (sec_pri_vza + sec_pri_sza)

    # calculate scattering angle
    pri_pha = cal_phaang(pri_sza, pri_vza, phi_rad)

    # main
    li_sp = overlap - sec_pri_vza - sec_pri_sza + 0.5 * (1 + np.cos(pri_pha)) * (sec_pri_vza * sec_pri_sza)
    return li_sp


if __name__ == '__main__':
    """
    the following script is for testing purpose only.
    """

    import matplotlib.pyplot as plt

    """
    # keep VZA, change RAZ
    # brf reaches minimum when RAZ=180, maximum when RAZ=0
    # 2pi cycle 
    """
    # for vza in range(10, 81, 10):
    #  ref = []
    #  for raz in range(-180, 181, 10):
    #    ref.append( brf_forward([45, vza, raz], [0.2442, 0.0047, 0.0550]) )
    #  plt.plot(ref, label='{}'.format(vza))
    # plt.legend()
    # plt.show()

    """
    # keep RAZ, change VZA
    #
    #
    """
    # ref = []
    # for vza in range(-90, 91):
    #   if vza <= 0:
    #       ref.append( brf_forward([30, -1*vza, 0], [0.0399, 0.0245, 0.0072]) )
    #   else:
    #       ref.append( brf_forward([30, vza, 180], [0.0399, 0.0245, 0.0072]) )
    # ref = np.array(ref)
    # # np.place(ref, ref<0, 0)
    # plt.plot(ref)
    # plt.ylim(0, 0.06)
    # plt.xlim(0, 180)
    # plt.xticks(range(0, 181, 15), range(-90, 91, 15))
    # plt.show()

    """
    """
    ref = []
    VZA = range(0, 81, 1)
    RAZ = range(0, 361, 5)
    for vza in VZA:
        tmp = []
        for raz in RAZ:
            tmp.append(brf_forward([29.179, vza, raz], [0.10251805, 0.188450884, 0.020178157]))

        ref.append(tmp)
    ref = np.array(ref)
    np.place(ref, ref > 1, 1)
    np.place(ref, ref < 0, 0)

    r = np.linspace(0, 80, len(VZA))
    theta = np.radians(np.linspace(0, 360, len(RAZ)))
    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("W")
    ax.set_rmax(80)
    ax.set_yticks([0, 30, 60, 80])
    # ax.set_yticklabels(map(str, range(0, , -10)))
    img1 = ax.contourf(theta, r, ref.transpose(), 40, cmap='jet', vmin=0, vmax=.15)
    cbar = fig.colorbar(img1)
    plt.show()
