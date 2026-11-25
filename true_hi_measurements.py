import numpy as np
import pandas as pd
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

# 1. Locate bright blind peaks (> 7.5 sigma) in part1 & part2
for part in ["part1", "part2"]:
    d_cube, w_cube, f_cube = cubes[part]
    for ch, plane in enumerate(d_cube):
        if np.all(np.isnan(plane)):
            continue
        pk = np.nanmax(plane) * 1000.0
        rms = np.nanstd(plane) * 1000.0
        if rms > 0 and (pk / rms) > 7.5:
            iy, ix = np.unravel_index(np.nanargmax(plane), plane.shape)
            sky = w_cube.pixel_to_world(ix, iy)
            vel_ch = c_kms * (f_rest / f_cube[ch] - 1.0)
            print(f"Blind Peak: {part} ch={ch+1} ({f_cube[ch]:.5f} GHz, cz={vel_ch:.0f} km/s) | Peak={pk:.2f} mJy, RMS={rms:.2f} mJy (SNR={pk/rms:.1f}) at {sky.to_string('hmsdms')}")

# 2. Extract unbiased zero-baseline spatial-mean spectra for all 19 in-field 6dFGS galaxies
rows = []
for _, row in df.iterrows():
    part = row["Cube"]
    data, wcs, freqs = cubes[part]
    nchan, ny, nx = data.shape
    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(c)
    ix, iy = int(round(float(x_pix))), int(round(float(y_pix)))
    if not (3 <= ix < nx - 3 and 3 <= iy < ny - 3):
        continue

    # Spatial mean over central 3x3 pixels (zero-baseline estimator)
    spec = np.nanmean(data[:, iy-1:iy+2, ix-1:ix+2], axis=(1, 2)) * 1000.0
    if np.all(np.isnan(spec)):
        continue

    vels = c_kms * (f_rest / freqs - 1.0)
    off_line = np.abs(vels - row["cz"]) > 600
    rms = np.nanstd(spec[off_line])

    in_line = np.abs(vels - row["cz"]) <= 450
    pk_flux = np.nanmax(spec[in_line])
    pk_vel = vels[in_line][np.nanargmax(spec[in_line])]
    snr = pk_flux / rms if rms > 0 else 0.0

    # Integrated flux S_int (Jy km/s) across positive channels within +/- 450 km/s
    dv = np.abs(np.median(np.diff(vels)))
    pos_chans = in_line & (spec > 0)
    s_int = np.nansum(spec[pos_chans]) * dv / 1000.0

    # Luminosity distance D_L (Mpc) for flat LCDM (H0=70, Om=0.3) and H I mass
    z = row["z"]
    d_l = (c_kms / 70.0) * z * (1.0 + 0.5 * (1.0 - 0.3) * z)
    m_hi = 2.356e5 * (d_l ** 2) * max(s_int, 0)
    log_mhi = np.log10(m_hi) if m_hi > 0 else np.nan
    m_hi_3sig = 2.356e5 * (d_l ** 2) * (3.0 * rms / 1000.0 * 300.0)
    log_mhi_3sig = np.log10(m_hi_3sig)

    status = "DETECTION" if snr >= 4.0 else ("CANDIDATE" if snr >= 3.0 else "UPPER_LIMIT")
    rows.append((row["name"], row["RA_hms"], row["DEC_dms"], int(row["cz"]), int(round(pk_vel)), round(rms, 2), round(pk_flux, 2), round(snr, 2), round(s_int, 3), round(log_mhi, 2), round(log_mhi_3sig, 2), status))

out_df = pd.DataFrame(rows, columns=["Galaxy", "RA_hms", "DEC_dms", "cz_opt", "cz_HI", "RMS_mJy", "Peak_mJy", "SNR", "Sint_Jy_kms", "logMHI", "logMHI_3sig_lim", "Status"])
out_df = out_df.sort_values("SNR", ascending=False)
print("\n=== Unbiased Zero-Baseline H I Measurements (All 19 In-Field Filament Galaxies) ===")
print(out_df.to_string(index=False))
out_df.to_csv("zone1_hi_final_catalog.csv", index=False)
