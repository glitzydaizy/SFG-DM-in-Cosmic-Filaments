import os, subprocess
import numpy as np
import pandas as pd

df = pd.read_csv("zone1_optical_filament_galaxies.csv")
os.makedirs("miriad_working/c2767_combined/mosaic_hi/spectra_z1", exist_ok=True)

results = []
for idx, row in df.iterrows():
    name = row["name"].replace(" ", "_")
    ra_hms = row["RA_hms"]
    dec_dms = row["DEC_dms"]
    f_exp = row["HI_Freq_GHz"]
    cube = f"miriad_working/c2767_combined/mosaic_hi/z1_comb_hi_{row['Cube']}.restor"
    log_file = f"miriad_working/c2767_combined/mosaic_hi/spectra_z1/{name}.txt"

    cmd = [
        "imspec",
        f"in={cube}",
        f"region=arcsec,box(-5,-5,5,5)({ra_hms},{dec_dms})",
        "options=mean",
        f"log={log_file}"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.exists(log_file):
        results.append((row["name"], row["cz"], f_exp, row["Cube"], np.nan, np.nan, np.nan, "OUTSIDE_FOV"))
        continue

    freqs, fluxes = [], []
    with open(log_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    chan = int(parts[0])
                    freq = float(parts[1])
                    flux = float(parts[2])
                    freqs.append(freq)
                    fluxes.append(flux)
                except ValueError:
                    continue

    freqs = np.array(freqs)
    fluxes = np.array(fluxes) * 1000.0  # mJy/beam
    if len(freqs) == 0:
        results.append((row["name"], row["cz"], f_exp, row["Cube"], np.nan, np.nan, np.nan, "NO_DATA"))
        continue

    rms = np.nanstd(fluxes)
    # Search within +/- 0.0025 GHz (~ +/- 550 km/s) of optical redshift
    win = np.abs(freqs - f_exp) <= 0.0025
    if np.any(win):
        peak_flux = np.nanmax(fluxes[win])
        peak_freq = freqs[win][np.nanargmax(fluxes[win])]
        snr = peak_flux / rms if rms > 0 else 0
    else:
        peak_flux, peak_freq, snr = np.nan, np.nan, np.nan

    status = "CANDIDATE_HI" if snr >= 3.5 else ("MARGINAL" if snr >= 2.8 else "NON_DET")
    results.append((row["name"], row["cz"], f_exp, row["Cube"], round(peak_freq, 5), round(peak_flux, 2), round(snr, 2), status))

res_df = pd.DataFrame(results, columns=["Galaxy", "cz_opt", "f_exp_GHz", "Cube", "f_peak_GHz", "Peak_mJy", "SNR", "Status"])
res_df = res_df.sort_values("SNR", ascending=False)
print("\n=== Zone 1 H I Cross-Match Results (Ranked by SNR) ===")
print(res_df.to_string(index=False))
res_df.to_csv("zone1_hi_crossmatch_summary.csv", index=False)
