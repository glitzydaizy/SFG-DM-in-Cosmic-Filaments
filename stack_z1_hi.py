import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

hi_dir = "miriad_working/c2767_combined/mosaic_hi"
df = pd.read_csv("zone1_optical_filament_galaxies.csv")
f_rest = 1.420405752
c_kms = 299792.458

cubes = {}
for part in ["part1", "part2"]:
    with fits.open(f"{hi_dir}/z1_comb_hi_{part}.fits") as hdul:
        primary = next(iter(hdul))
        data = np.squeeze(primary.data)
        hdr = primary.header
    wcs = WCS(hdr).celestial
    nchan = len(data)
    freqs = hdr["CRVAL3"] + (np.arange(nchan) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    if np.nanmean(freqs) > 1e6:
        freqs = freqs / 1e9
    cubes[part] = (data, wcs, freqs)

# Velocity grid in rest frame: -2500 to +2500 km/s in steps of 225 km/s
v_grid = np.arange(-2475, 2476, 225.0)
aligned_fluxes = []
aligned_masses = []
weights = []
used_names = []

for _, row in df.iterrows():
    part = row["Cube"]
    data, wcs, freqs = cubes[part]
    nchan, ny, nx = data.shape
    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(c)
    ix, iy = int(round(float(x_pix))), int(round(float(y_pix)))
    if not (2 <= ix < nx - 2 and 2 <= iy < ny - 2):
        continue

    spec_mJy = data[:, iy, ix] * 1000.0
    if np.all(np.isnan(spec_mJy)):
        continue

    vels = c_kms * (f_rest / freqs - 1.0)
    dv_rest = (vels - row["cz"]) / (1.0 + row["z"])
    off_line = np.abs(dv_rest) > 600
    rms = np.nanstd(spec_mJy[off_line])
    if rms <= 0 or np.isnan(rms):
        continue

    # Interpolate onto common rest-frame velocity grid (fill out-of-band with NaN)
    order = np.argsort(dv_rest)
    interp_flux = np.interp(v_grid, dv_rest[order], spec_mJy[order], left=np.nan, right=np.nan)
    z = row["z"]
    d_l = (c_kms / 70.0) * z * (1.0 + 0.5 * (1.0 - 0.3) * z)
    # H I mass per unit velocity (M_sun / (km/s))
    interp_mass_dens = 2.356e5 * (d_l ** 2) * (interp_flux / 1000.0)

    aligned_fluxes.append(interp_flux)
    aligned_masses.append(interp_mass_dens)
    weights.append(1.0 / (rms ** 2))
    used_names.append(row["name"])

aligned_fluxes = np.array(aligned_fluxes)
aligned_masses = np.array(aligned_masses)
weights = np.array(weights)[:, None]

# Inverse-variance weighted stack
valid = ~np.isnan(aligned_fluxes)
w_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
stack_flux = np.sum(np.where(valid, aligned_fluxes * weights, 0.0), axis=0) / w_sum
stack_mass_dens = np.sum(np.where(valid, aligned_masses * weights, 0.0), axis=0) / w_sum

off = np.abs(v_grid) > 600
stack_rms = np.nanstd(stack_flux[off])
center_mask = np.abs(v_grid) <= 300
pk_stack = np.nanmax(stack_flux[center_mask])
snr_stack = pk_stack / stack_rms

dv_grid = 225.0
mean_mhi = np.nansum(stack_mass_dens[center_mask]) * dv_grid

print(f"Stacked {len(used_names)} Zone 1 filament galaxies:")
print(f"  -> Stacked Off-Line RMS : {stack_rms:.3f} mJy/beam (down from ~1.3 mJy/beam!)")
print(f"  -> Stacked Peak Flux    : {pk_stack:.3f} mJy/beam (SNR = {snr_stack:.2f} sigma)")
print(f"  -> Average H I Mass     : <M_HI> = {mean_mhi:.2e} M_sun (log10 <M_HI> = {np.log10(mean_mhi):.2f})")

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.step(v_grid, stack_flux, where="mid", color="navy", lw=1.8, label=f"Stacked H I Spectrum (N = {len(used_names)} galaxies)")
ax.axhline(0, color="gray", ls=":", lw=1.0)
ax.axhline(3 * stack_rms, color="orange", ls="--", lw=1.1, label=f"3σ Stacked Noise ({3*stack_rms:.2f} mJy/beam)")
ax.axvline(0, color="crimson", ls="--", lw=1.3, label="Rest-Frame Optical Center (Δv = 0 km/s)")
ax.axvspan(-300, 300, color="royalblue", alpha=0.12, label="H I Integration Window (±300 km/s)")
ax.set_xlabel("Rest-Frame Velocity Offset Δv (km/s)", fontsize=11)
ax.set_ylabel("Weighted Mean Flux Density (mJy/beam)", fontsize=11)
ax.set_title(f"Zone 1 Filament Stacked 21-cm H I Spectrum (Peak SNR = {snr_stack:.1f}σ, log <M_HI> = {np.log10(mean_mhi):.2f})", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("zone1_hi_stacked_spectrum.png")
print("Saved stacked spectrum figure to: zone1_hi_stacked_spectrum.png")
