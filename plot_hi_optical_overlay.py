import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.skyview import SkyView

orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
top = orig[orig["name"] == "6dFGS gJ033929.2-535120"].squeeze()
c_top = SkyCoord(float(top["ra"]) * u.deg, float(top["dec"]) * u.deg)
print(f"Fetching DSS2 Red optical cutout for {top['name']} ({c_top.to_string('hmsdms')})...")

opt_list = SkyView.get_images(position=c_top, survey=["DSS2 Red"], radius=1.25 * u.arcmin)
opt_hdul = next(iter(opt_list))
opt_primary = next(iter(opt_hdul))
opt_data = opt_primary.data
opt_wcs = WCS(opt_primary.header)

hi_dir = "miriad_working/c2767_4blocks_hi"
with fits.open(f"{hi_dir}/z1_4blk_hi_part1.fits") as hdul:
    primary = next(iter(hdul))
    hi_data = np.squeeze(primary.data)
    hi_hdr = primary.header

hi_wcs = WCS(hi_hdr).celestial
c_kms = 299792.458
f_rest = 1.420405752
freqs = hi_hdr["CRVAL3"] + (np.arange(len(hi_data)) + 1 - hi_hdr["CRPIX3"]) * hi_hdr["CDELT3"]
if np.nanmean(freqs) > 1e6:
    freqs = freqs / 1e9
vels = c_kms * (f_rest / freqs - 1.0)
dv = np.abs(np.median(np.diff(vels)))

ch_mask = np.abs(vels - float(top["cz"])) <= 350
mom0_full = np.nansum(hi_data[ch_mask], axis=0) * dv

ix0, iy0 = [int(round(float(x))) for x in hi_wcs.world_to_pixel(c_top)]
cutout_hi = mom0_full[iy0-30:iy0+31, ix0-30:ix0+31]
mask_bg = np.ones_like(cutout_hi, dtype=bool)
mask_bg[25:36, 25:36] = False
mom0_rms = np.nanstd(cutout_hi[mask_bg])
print(f"Local Moment 0 RMS = {mom0_rms:.3f} Jy/beam km/s | Peak near center = {np.nanmax(cutout_hi[25:36, 25:36]):.3f} Jy/beam km/s")

fig = plt.figure(figsize=(7.5, 7), dpi=150)
ax = fig.add_subplot(111, projection=opt_wcs)

vmin, vmax = np.percentile(opt_data, [5, 99.5])
ax.imshow(opt_data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

levels = [2.0 * mom0_rms, 2.5 * mom0_rms, 3.0 * mom0_rms]
cs = ax.contour(
    mom0_full,
    levels=levels,
    colors=["dodgerblue", "cyan", "lime"],
    linewidths=[1.5, 1.8, 2.0],
    transform=ax.get_transform(hi_wcs),
)
ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f"{x/mom0_rms:.1f}σ")

ax.scatter(
    [c_top.ra.deg], [c_top.dec.deg],
    transform=ax.get_transform("world"),
    marker="+", s=160, color="crimson", lw=2.0, zorder=6,
    label="6dFGS Optical Center (cz = 17,520 km/s)"
)

pix_scale_deg = abs(opt_primary.header["CDELT2"])
bmaj_pix = (14.346 / 3600.0) / pix_scale_deg
bmin_pix = (7.966 / 3600.0) / pix_scale_deg
ny_opt, nx_opt = opt_data.shape
beam = Ellipse(
    xy=(0.14 * nx_opt, 0.12 * ny_opt),
    width=bmin_pix, height=bmaj_pix, angle=-41.3,
    facecolor="gold", edgecolor="black", alpha=0.85, zorder=7,
    label='ATCA Beam (14.3" × 8.0")'
)
ax.add_patch(beam)

ax.set_xlim(0, nx_opt - 1)
ax.set_ylim(0, ny_opt - 1)
ax.set_xlabel("Right Ascension (J2000)", fontsize=11)
ax.set_ylabel("Declination (J2000)", fontsize=11)
ax.set_title(f"DSS2 Optical Cutout + ATCA 21-cm H I Contours\n{top['name']} (log M_HI = 10.27 M☉)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")

plt.tight_layout()
plt.savefig("zone1_hi_optical_overlay.png")
print("Saved multiwavelength H I + Optical overlay to: zone1_hi_optical_overlay.png")
