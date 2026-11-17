import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.table import Table

# Standard flat LambdaCDM cosmology (H0 = 70 km/s/Mpc, Om0 = 0.3)
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

# Mean observing frequency of d1213_mfs.cm (GHz) and map RMS (mJy)
nu_obs_GHz = 2.457  # Check exact CRVAL3 from imhead if needed (~2.15-2.5 GHz)
rms_mJy = 0.0896   # 89.6 uJy/beam from imstat
alpha = 0.75       # Standard non-thermal synchrotron spectral index (S ~ nu^-alpha)

# Confirmed Horologium-Reticulum Star-Forming Galaxies from d1213_mfs_catalog.csv
sfg_sample = [
    {
        "ID": "ATCA_18_0337-5345",
        "WISEA": "WISEA J033706.65-534512.5",
        "z": 0.075542,
        "S_int_mJy": 4.152,
        "q22": 0.53,
        "W1_W2": 0.234
    },
    {
        "ID": "ATCA_20_0335-5345",
        "WISEA": "WISEA J033555.94-534510.8",
        "z": 0.059641,
        "S_int_mJy": 1.789,
        "q22": 1.03,
        "W1_W2": 0.184
    }
]

results = []
for g in sfg_sample:
    z = g["z"]
    d_L = cosmo.luminosity_distance(z).to(u.m).value  # meters

    # 1. Scale observed flux density to rest-frame 1.4 GHz:
    # S_1.4 = S_obs * (1.4 / nu_obs)^(-alpha) * (1 + z)^(alpha - 1)
    k_corr = (1.0 + z) ** (alpha - 1.0)
    freq_scale = (1.4 / nu_obs_GHz) ** (-alpha)
    S_14_mJy = g["S_int_mJy"] * freq_scale * k_corr
    e_S_14_mJy = np.sqrt(rms_mJy**2 + (0.05 * g["S_int_mJy"])**2) * freq_scale * k_corr  # RMS + 5% cal error

    # 2. Rest-frame 1.4 GHz monochromatic luminosity (W / Hz)
    S_14_W_m2_Hz = S_14_mJy * 1e-29  # 1 mJy = 1e-29 W m^-2 Hz^-1
    L_14 = 4.0 * np.pi * (d_L ** 2) * S_14_W_m2_Hz
    e_L_14 = L_14 * (e_S_14_mJy / S_14_mJy)

    # 3. Star-Formation Rate calibrations (M_sun / yr)
    # Murphy et al. (2011): SFR_1.4GHz = 6.35e-29 * L_1.4 (erg/s/Hz) = 6.35e-22 * L_1.4 (W/Hz)
    sfr_murphy = 6.35e-22 * L_14
    e_sfr_murphy = 6.35e-22 * e_L_14

    # Bell (2003): SFR = 5.52e-22 * L_1.4 (W/Hz) for L > 6.4e21 W/Hz
    sfr_bell = 5.52e-22 * L_14
    e_sfr_bell = 5.52e-22 * e_L_14

    results.append((
        g["ID"], g["WISEA"], round(z, 6), round(cosmo.luminosity_distance(z).value, 1),
        round(g["S_int_mJy"], 3), round(S_14_mJy, 3), round(e_S_14_mJy, 3),
        f"{L_14:.2e}", round(sfr_murphy, 2), round(e_sfr_murphy, 2),
        round(sfr_bell, 2), round(e_sfr_bell, 2)
    ))

tbl = Table(
    rows=results,
    names=["ID", "WISEA", "z", "D_L_Mpc", "S_obs_mJy", "S_1.4_mJy", "e_S_1.4",
           "L_1.4_W_Hz", "SFR_Murphy11", "e_SFR_M11", "SFR_Bell03", "e_SFR_B03"]
)
tbl.pprint(max_width=-1)
tbl.write("results/d1213_sfg_sfr.csv", format="csv", overwrite=True)