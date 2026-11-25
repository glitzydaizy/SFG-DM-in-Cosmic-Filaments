import os, glob, subprocess
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

hi_dir = "miriad_working/c2767_combined/mosaic_hi"
df = pd.read_csv("zone1_optical_filament_galaxies.csv")

# Locate or export FITS cubes for part1 and part2
for part in ["part1", "part2"]:
    fits_path = f"{hi_dir}/z1_comb_hi_{part}.fits"
    if not os.path.exists(fits_path):
        mir_matches = sorted(glob.glob(f"{hi_dir}/*{part}*.restor"))
        if not mir_matches:
            mir_matches = sorted(glob.glob(f"{hi_dir}/*{part}*"))
        print(f"Exporting {mir_matches[0]} -> {fits_path}")
        subprocess.run(["fits", f"in={mir_matches[0]}", f"out={fits_path}", "op=xyout"], check=True)

cubes = {}
for part in ["part1", "part2"]:
    hdul = fits.open(f"{hi_dir}/z1_comb_hi_{part}.fits")
    data = np.squeeze(hdul[0].data)  # shape: (nchan, ny, nx)
    wcs = WCS(hdul[0].header).celestial
    hdr = hdul[0].header
    # Build frequency axis in GHz
    nchan = data.shape[0]
    crval3 = hdr["CRVAL3"]
    cdelt3 = hdr["CDELT3"]
    crpix3 = hdr["CRPIX3"]
    freqs = crval3 + (np.arange(nchan) + 1 - crpix3) * cdelt3
    if np.nanmean(freqs) > 1e6:
        freqs = freqs / 1e9  # convert Hz -> GHz
    cubes[part] = (data, wcs, freqs)
    print(f"Loaded {part}: shape={data.shape}, freq={freqs.min():.5f} to {freqs.max():.5f} GHz")

os.makedirs(f"{hi_dir}/spectra_z1_csv", exist_ok=True)
results = []

for _, row in df.iterrows():
    part = row["Cube"]
    data, wcs, freqs = cubes[part]
    nchan, ny, nx = data.shape

    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(c)
    ix, iy = int(round(float(x_pix))), int(round(float(y_pix)))

    if not (2 <= ix < nx - 2 and 2 <= iy < ny - 2):
        results.append((row["name"], row["RA_hms"], row["DEC_dms"], row["cz"], row["HI_Freq_GHz"], part, np.nan, np.nan, np.nan, "OUTSIDE_IMAGE"))
        continue

    # Extract 3x3 pixel box around galaxy position (in mJy/beam)
    spec = np.nanmax(data[:, iy-1:iy+2, ix-1:ix+2], axis=(1, 2)) * 1000.0
    if np.all(np.isnan(spec)):
        results.append((row["name"], row["RA_hms"], row["DEC_dms"], row["cz"], row["HI_Freq_GHz"], part, np.nan, np.nan, np.nan, "MASKED_EDGE"))
        continue

    pd.DataFrame({"freq_GHz": freqs, "flux_mJy": spec}).to_csv(
        f"{hi_dir}/spectra_z1_csv/{row['name'].replace(' ', '_')}.csv", index=False
    )

    rms = np.nanstd(spec)
    win = np.abs(freqs - row["HI_Freq_GHz"]) <= 0.0025  # +/- 550 km/s window
    if np.any(win) and not np.all(np.isnan(spec[win])):
        pk_idx = np.nanargmax(spec[win])
        pk_flux = spec[win][pk_idx]
        pk_freq = freqs[win][pk_idx]
        snr = pk_flux / rms if rms > 0 else 0.0
    else:
        pk_flux, pk_freq, snr = np.nan, np.nan, np.nan

    status = "DETECTION" if snr >= 4.0 else ("CANDIDATE" if snr >= 3.0 else "NON_DET")
    results.append((row["name"], row["RA_hms"], row["DEC_dms"], row["cz"], row["HI_Freq_GHz"], part, round(pk_freq, 5), round(pk_flux, 2), round(snr, 2), status))

res_df = pd.DataFrame(results, columns=["Galaxy", "RA_hms", "DEC_dms", "cz_opt", "f_exp_GHz", "Cube", "f_peak_GHz", "Peak_mJy", "SNR", "Status"])
res_df = res_df.sort_values("SNR", ascending=False)
print("\n=== Zone 1 H I Cross-Match Results (Astropy WCS Extraction) ===")
print(res_df.to_string(index=False))
res_df.to_csv("zone1_hi_crossmatch_summary.csv", index=False)
