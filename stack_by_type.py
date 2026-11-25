import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

df = pd.read_csv("zone1_hi_exact_beam_catalog.csv")
orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
df = df.merge(orig[["name", "ra", "dec", "z", "bj_mag"]], left_on="Galaxy", right_on="name", how="left")

# Query HyperLEDA (morphology T-type) and AllWISE (W1-W3 star formation color) via CfA mirror
v = Vizier(columns=["*"], row_limit=5, vizier_server="vizier.cfa.harvard.edu")
morph_types, w1_mags, w3_mags, w1_w3 = [], [], [], []

for _, row in df.iterrows():
    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)

    # HyperLEDA morphology
    r_pgc = v.query_region(c, radius=15 * u.arcsec, catalog="VII/237/pgc")
    if len(r_pgc) > 0 and len(r_pgc[0]) > 0 and "type" in r_pgc[0].colnames:
        morph_types.append(str(r_pgc[0][0]["type"]))
    else:
        morph_types.append("")

    # AllWISE photometry
    r_wise = v.query_region(c, radius=10 * u.arcsec, catalog="II/328/allwise")
    if len(r_wise) > 0 and len(r_wise[0]) > 0 and "W1mag" in r_wise[0].colnames and "W3mag" in r_wise[0].colnames:
        try:
            w1 = float(r_wise[0][0]["W1mag"])
            w3 = float(r_wise[0][0]["W3mag"])
            w1_mags.append(round(w1, 2))
            w3_mags.append(round(w3, 2))
            w1_w3.append(round(w1 - w3, 2))
        except Exception:
            w1_mags.append(np.nan)
            w3_mags.append(np.nan)
            w1_w3.append(np.nan)
    else:
        w1_mags.append(np.nan)
        w3_mags.append(np.nan)
        w1_w3.append(np.nan)

df["T_type"] = morph_types
df["W1mag"] = w1_mags
df["W1_W3"] = w1_w3
df["IR_Class"] = np.where(df["W1_W3"] > 1.5, "Star-Forming", np.where(df["W1_W3"].notna(), "Passive/Quenched", "Unknown"))

cols = ["Galaxy", "cz_opt", "SNR", "logMHI", "logMHI_3sig_lim", "Status", "T_type", "W1_W3", "IR_Class"]
print("=== Zone 1 Filament Sample with HyperLEDA Morphology & AllWISE Star-Formation Class ===")
print(df[cols].to_string(index=False))
df.to_csv("zone1_hi_multiwavelength_catalog.csv", index=False)

# Quick subset stack check: Star-Forming vs Passive/Quenched
hi_dir = "miriad_working/c2767_combined/mosaic_hi"
f_rest, c_kms = 1.420405752, 299792.458
cubes = {}
for part in ["part1", "part2"]:
    with fits.open(f"{hi_dir}/z1_comb_hi_{part}.fits") as hdul:
        primary = next(iter(hdul))
        data = np.squeeze(primary.data)
        hdr = primary.header
    wcs = WCS(hdr).celestial
    freqs = hdr["CRVAL3"] + (np.arange(len(data)) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    if np.nanmean(freqs) > 1e6:
        freqs = freqs / 1e9
    cubes[part] = (data, wcs, freqs)

v_grid = np.arange(-2475, 2476, 225.0)
print("\n=== Subset H I Stacking Summary ===")
for label in ["Star-Forming", "Passive/Quenched"]:
    sub = df[df["IR_Class"] == label]
    if len(sub) == 0:
        continue
    flx_list, m_list, w_list = [], [], []
    for _, row in sub.iterrows():
        part = "part1" if (f_rest / (1.0 + row["z"])) >= 1.332 else "part2"
        data, wcs, freqs = cubes[part]
        ix, iy = [int(round(float(x))) for x in wcs.world_to_pixel(SkyCoord(row["ra"]*u.deg, row["dec"]*u.deg))]
        spec = data[:, iy, ix] * 1000.0
        dv_rest = (c_kms * (f_rest / freqs - 1.0) - row["cz_opt"]) / (1.0 + row["z"])
        rms = np.nanstd(spec[np.abs(dv_rest) > 600])
        order = np.argsort(dv_rest)
        interp_f = np.interp(v_grid, dv_rest[order], spec[order], left=np.nan, right=np.nan)
        d_l = (c_kms / 70.0) * row["z"] * (1.0 + 0.35 * row["z"])
        flx_list.append(interp_f)
        m_list.append(2.356e5 * (d_l ** 2) * (interp_f / 1000.0))
        w_list.append(1.0 / (rms ** 2))
    flx_arr, m_arr, w_arr = np.array(flx_list), np.array(m_list), np.array(w_list)[:, None]
    valid = ~np.isnan(flx_arr)
    w_sum = np.sum(np.where(valid, w_arr, 0.0), axis=0)
    st_f = np.sum(np.where(valid, flx_arr * w_arr, 0.0), axis=0) / w_sum
    st_m = np.sum(np.where(valid, m_arr * w_arr, 0.0), axis=0) / w_sum
    st_rms = np.nanstd(st_f[np.abs(v_grid) > 600])
    cmask = np.abs(v_grid) <= 300
    pk = np.nanmax(st_f[cmask])
    mean_m = np.nansum(st_m[cmask]) * 225.0
    print(f"  {label:17s} (N={len(sub):2d}): RMS={st_rms:.3f} mJy | Peak={pk:.3f} mJy (SNR={pk/st_rms:.2f}σ) | <M_HI>={mean_m:.2e} M_sun")
