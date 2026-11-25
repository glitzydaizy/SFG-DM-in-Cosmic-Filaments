import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Load environment & cube data for Core (D_fil < 1.0 Mpc) vs Outskirts (D_fil >= 1.0 Mpc) stacking
df = pd.read_csv("zone1_filament_environment.csv")
hi_dir = "miriad_working/c2767_4blocks_hi"
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

v_grid = np.arange(-2475, 2476, 225.0)
core_sub = df[df["D_fil_Mpc"] < 1.0]
out_sub = df[df["D_fil_Mpc"] >= 1.0]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
print("=== Inner Core vs. Outer Filament H I Stacking (cf. Kleiner+17; Crone Odekon+18) ===")

for ax, (label, sub, col) in zip(axes, [
    ("Filament Core (D_fil < 1.0 Mpc)", core_sub, "darkmagenta"),
    ("Filament Outskirts (D_fil ≥ 1.0 Mpc)", out_sub, "teal")
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
    st_f = np.sum(np.where(valid, flx_arr * w_arr, 0.0), axis=0) / w_sum
    st_m = np.sum(np.where(valid, m_arr * w_arr, 0.0), axis=0) / w_sum

    st_rms = np.nanstd(st_f[np.abs(v_grid) > 600])
    cmask = np.abs(v_grid) <= 300
    pk = np.nanmax(st_f[cmask])
    snr_st = pk / st_rms
    mean_m = np.nansum(st_m[cmask]) * 225.0
    print(f"  {label:38s} (N={len(sub):2d}): RMS={st_rms:.3f} mJy | Peak={pk:.3f} mJy (SNR={snr_st:.2f}σ) | <M_HI>={mean_m:.2e} M_sun")

    ax.step(v_grid, st_f, where="mid", color=col, lw=1.7)
    ax.axhline(0, color="gray", ls=":", lw=0.9)
    ax.axhline(3 * st_rms, color="orange", ls="--", lw=1.0, label=f"3σ ({3*st_rms:.2f} mJy)")
    ax.axvline(0, color="crimson", ls="--", lw=1.1)
    ax.axvspan(-300, 300, color=col, alpha=0.10)
    ax.set_title(f"{label} (N={len(sub)})\nSNR={snr_st:.2f}σ | <M_HI>={mean_m:.1e} M☉", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rest-Frame Δv (km/s)", fontsize=9.5)
    ax.set_ylabel("Mean Flux (mJy/beam)", fontsize=9.5)
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig("zone1_inner_vs_outer_stack.png")
print("Saved: zone1_inner_vs_outer_stack.png")

# 2. Stellar-to-Halo Mass Relation (SHMR) vs. Moster et al. (2013)
log_m200 = np.linspace(11.0, 13.5, 200)
m200 = 10.0 ** log_m200
# Moster et al. (2013) z~0 parameters: log M1 = 11.590, N = 0.0351, beta = 1.376, gamma = 0.608
M1 = 10.0 ** 11.590
N_norm = 0.0351
beta, gamma = 1.376, 0.608
shmr_ratio = 2.0 * N_norm / ((m200 / M1) ** (-beta) + (m200 / M1) ** gamma)
log_mstar_moster = np.log10(shmr_ratio * m200)

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
ax.plot(log_m200, log_mstar_moster, color="black", lw=2.2, label="Moster et al. (2013) ΛCDM Abundance Matching")
ax.fill_between(log_m200, log_mstar_moster - 0.15, log_mstar_moster + 0.15, color="gray", alpha=0.2, label="±0.15 dex Intrinsic Scatter")

# Plot Zone 1 Star-Forming Filament Ensemble: log M_200 = 12.38, log M_* = 10.76
ax.errorbar([12.38], [10.76], xerr=[0.18], yerr=[0.12], fmt="D", color="crimson", markersize=9, zorder=5,
            label="Zone 1 SF Filament Ensemble (NFW Fit)")
ax.set_xlabel("log10(M_200 / M☉) [Dark Matter Halo Virial Mass]", fontsize=10.5)
ax.set_ylabel("log10(M_* / M☉) [Stellar Mass]", fontsize=10.5)
ax.set_title("Stellar-to-Halo Mass Relation: Zone 1 Filament vs. ΛCDM Prediction", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("zone1_shmr_comparison.png")
print("Saved: zone1_shmr_comparison.png")
