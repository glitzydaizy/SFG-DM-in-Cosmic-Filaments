import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

hi_dir = "miriad_working/c2767_4blocks_hi"
df = pd.read_csv("zone1_hi_multiwavelength_catalog.csv")
orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
df = df.drop(columns=[c for c in ["ra","dec","z","RA_hms","DEC_dms"] if c in df.columns]).merge(orig[["name","ra","dec","z","RA_hms","DEC_dms"]], left_on="Galaxy", right_on="name", how="left")

f_rest, c_kms = 1.420405752, 299792.458
cubes = {}
for part in ["part1", "part2"]:
    with fits.open(f"{hi_dir}/z1_4blk_hi_{part}.fits") as hdul:
        primary = next(iter(hdul))
        data = np.squeeze(primary.data)
        hdr = primary.header
    wcs = WCS(hdr).celestial
    freqs = hdr["CRVAL3"] + (np.arange(len(data)) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    if np.nanmean(freqs) > 1e6:
        freqs = freqs / 1e9
    cubes[part] = (data, wcs, freqs)

rows = []
for _, row in df.iterrows():
    part = "part1" if (f_rest / (1.0 + row["z"])) >= 1.332 else "part2"
    data, wcs, freqs = cubes[part]
    ix, iy = [int(round(float(x))) for x in wcs.world_to_pixel(SkyCoord(row["ra"]*u.deg, row["dec"]*u.deg))]
    spec = data[:, iy, ix] * 1000.0
    vels = c_kms * (f_rest / freqs - 1.0)

    rms = np.nanstd(spec[np.abs(vels - row["cz_opt"]) > 600])
    in_line = np.abs(vels - row["cz_opt"]) <= 450
    pk_flux = np.nanmax(spec[in_line])
    pk_vel = vels[in_line][np.nanargmax(spec[in_line])]
    snr = pk_flux / rms if rms > 0 else 0.0

    dv = np.abs(np.median(np.diff(vels)))
    s_int = np.nansum(spec[in_line & (spec > 0)]) * dv / 1000.0
    d_l = (c_kms / 70.0) * row["z"] * (1.0 + 0.35 * row["z"])
    m_hi = 2.356e5 * (d_l ** 2) * max(s_int, 0)
    log_mhi = round(np.log10(m_hi), 2) if m_hi > 0 else np.nan
    log_lim = round(np.log10(2.356e5 * (d_l ** 2) * (3.0 * rms / 1000.0 * 300.0)), 2)

    status = "DETECTION" if snr >= 3.5 else ("CANDIDATE" if snr >= 2.8 else "UPPER_LIMIT")
    rows.append((row["Galaxy"], row["RA_hms"], row["DEC_dms"], int(row["cz_opt"]), int(round(pk_vel)),
                 round(rms, 2), round(pk_flux, 2), round(snr, 2), round(s_int, 3), log_mhi, log_lim, status, row["IR_Class"]))

cat_df = pd.DataFrame(rows, columns=["Galaxy", "RA_hms", "DEC_dms", "cz_opt", "cz_HI", "RMS_mJy", "Peak_mJy", "SNR", "Sint_Jy_kms", "logMHI", "logMHI_3sig_lim", "Status", "IR_Class"])
cat_df = cat_df.sort_values("SNR", ascending=False)
print("=== Full-Depth 4-Block Zone 1 H I Catalog (N = 19) ===")
print(cat_df.to_string(index=False))
cat_df.to_csv("zone1_4blk_hi_catalog.csv", index=False)

# Stacking across All (N=19), Star-Forming (N=9), and Passive/Quenched (N=7)
v_grid = np.arange(-2475, 2476, 225.0)
print("\n=== Full-Depth 4-Block H I Stacking Summary ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

for ax, (label, sub) in zip(axes, [
    ("All Filament Galaxies", df),
    ("Star-Forming (W1-W3 > 1.5)", df[df["IR_Class"] == "Star-Forming"]),
    ("Passive/Quenched (W1-W3 <= 1.5)", df[df["IR_Class"] == "Passive/Quenched"])
]):
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
    st_f = np.divide(np.sum(np.where(valid, flx_arr * w_arr, 0.0), axis=0), w_sum, out=np.full_like(v_grid, np.nan), where=w_sum > 0)
    st_m = np.divide(np.sum(np.where(valid, m_arr * w_arr, 0.0), axis=0), w_sum, out=np.full_like(v_grid, np.nan), where=w_sum > 0)

    st_rms = np.nanstd(st_f[np.abs(v_grid) > 600])
    cmask = np.abs(v_grid) <= 300
    pk = np.nanmax(st_f[cmask])
    snr_st = pk / st_rms
    mean_m = np.nansum(st_m[cmask]) * 225.0
    print(f"  {label:32s} (N={len(sub):2d}): RMS={st_rms:.3f} mJy | Peak={pk:.3f} mJy (SNR={snr_st:.2f}σ) | <M_HI>={mean_m:.2e} M_sun")

    ax.step(v_grid, st_f, where="mid", color="navy", lw=1.6)
    ax.axhline(0, color="gray", ls=":", lw=0.9)
    ax.axhline(3 * st_rms, color="orange", ls="--", lw=1.0, label=f"3σ ({3*st_rms:.2f} mJy)")
    ax.axvline(0, color="crimson", ls="--", lw=1.1)
    ax.axvspan(-300, 300, color="royalblue", alpha=0.12)
    ax.set_title(f"{label} (N={len(sub)})\nSNR={snr_st:.2f}σ | <M_HI>={mean_m:.1e} M☉", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Rest-Frame Δv (km/s)", fontsize=9)
    ax.set_ylabel("Mean Flux (mJy/beam)", fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("zone1_4blk_stacked_spectrum.png")
print("\nSaved 3-panel stacked spectrum plot to: zone1_4blk_stacked_spectrum.png")
