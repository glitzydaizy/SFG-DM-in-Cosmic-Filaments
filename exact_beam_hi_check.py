import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

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

# 1. Exact central-pixel (Jy/beam) extraction for all 19 in-field 6dFGS galaxies
rows = []
for _, row in df.iterrows():
    part = row["Cube"]
    data, wcs, freqs = cubes[part]
    nchan, ny, nx = data.shape
    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(c)
    ix, iy = int(round(float(x_pix))), int(round(float(y_pix)))
    if not (2 <= ix < nx - 2 and 2 <= iy < ny - 2):
        continue

    # Exact central pixel in mJy/beam (true beam-weighted flux at galaxy position)
    spec = data[:, iy, ix] * 1000.0
    if np.all(np.isnan(spec)):
        continue

    vels = c_kms * (f_rest / freqs - 1.0)
    off_line = np.abs(vels - row["cz"]) > 600
    rms = np.nanstd(spec[off_line])

    in_line = np.abs(vels - row["cz"]) <= 450
    pk_flux = np.nanmax(spec[in_line])
    pk_vel = vels[in_line][np.nanargmax(spec[in_line])]
    snr = pk_flux / rms if rms > 0 else 0.0

    dv = np.abs(np.median(np.diff(vels)))
    pos_chans = in_line & (spec > 0)
    s_int = np.nansum(spec[pos_chans]) * dv / 1000.0  # Jy km/s

    z = row["z"]
    d_l = (c_kms / 70.0) * z * (1.0 + 0.5 * (1.0 - 0.3) * z)
    m_hi = 2.356e5 * (d_l ** 2) * max(s_int, 0)
    log_mhi = np.log10(m_hi) if m_hi > 0 else np.nan
    m_hi_3sig = 2.356e5 * (d_l ** 2) * (3.0 * rms / 1000.0 * 300.0)
    log_mhi_3sig = np.log10(m_hi_3sig)

    status = "DETECTION" if snr >= 3.5 else ("CANDIDATE" if snr >= 2.8 else "UPPER_LIMIT")
    rows.append((row["name"], row["RA_hms"], row["DEC_dms"], int(row["cz"]), int(round(pk_vel)), round(rms, 2), round(pk_flux, 2), round(snr, 2), round(s_int, 3), round(log_mhi, 2), round(log_mhi_3sig, 2), status))

out_df = pd.DataFrame(rows, columns=["Galaxy", "RA_hms", "DEC_dms", "cz_opt", "cz_HI", "RMS_mJy", "Peak_mJy", "SNR", "Sint_Jy_kms", "logMHI", "logMHI_3sig_lim", "Status"])
out_df = out_df.sort_values("SNR", ascending=False)
print("=== Exact Single-Beam (Central Pixel) H I Measurements ===")
print(out_df.to_string(index=False))
out_df.to_csv("zone1_hi_exact_beam_catalog.csv", index=False)

# 2. Check HyperLEDA & Fleenor 2005 for any extra galaxies in 13,000 - 26,500 km/s
center = SkyCoord(54.125 * u.deg, -53.3333 * u.deg)
v = Vizier(columns=["*"], row_limit=500, vizier_server="vizier.cfa.harvard.edu")
res = v.query_region(center, radius=0.75 * u.deg, catalog=["J/AJ/130/957", "VII/237/pgc"])

if "VII/237/pgc" in res:
    pgc = res["VII/237/pgc"].to_pandas()
    if "cz" in pgc.columns:
        pgc_fil = pgc[(pgc["cz"] >= 13000) & (pgc["cz"] <= 26500)].sort_values("cz")
        print(f"\n=== HyperLEDA Galaxies in 13,000-26,500 km/s ({len(pgc_fil)} rows) ===")
        cols = [c for c in ["PGC", "Name", "RAJ2000", "DEJ2000", "cz", "e_cz", "type", "logd25"] if c in pgc_fil.columns]
        print(pgc_fil[cols].to_string(index=False))

if "J/AJ/130/957/table2" in res:
    print("\n=== Fleenor et al. 2005 HRS Table 2 ===")
    print(res["J/AJ/130/957/table2"].to_pandas().to_string(index=False))
