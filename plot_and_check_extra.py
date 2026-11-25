import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# 1. Safely check any extra CSVs or query Fleenor2005 + HyperLEDA via CfA Harvard mirror
print("=== Checking extra catalogs via CfA Harvard VizieR ===")
center = SkyCoord(54.125 * u.deg, -53.3333 * u.deg)
v = Vizier(columns=["*"], row_limit=500, vizier_server="vizier.cfa.harvard.edu")
res = v.query_region(center, radius=0.75 * u.deg, catalog=["J/AJ/130/957", "VII/237/pgc"])

for k in res.keys():
    df_extra = res[k].to_pandas()
    print(f"\nCatalog {k}: {len(df_extra)} rows")
    vel_cols = [c for c in df_extra.columns if any(x in c.lower() for x in ["v", "cz", "z"])]
    print("Velocity/redshift columns:", vel_cols)
    for vc in vel_cols:
        if pd.api.types.is_numeric_dtype(df_extra[vc]):
            sub = df_extra[(df_extra[vc] >= 13000) & (df_extra[vc] <= 26500)]
            if len(sub) > 0:
                print(f"Rows in 13,000-26,500 km/s ({vc}): {len(sub)}")
                print(sub.to_string(index=False))

# 2. Plot 1D H I spectra for the top 8 H I detections/candidates
hi_dir = "miriad_working/c2767_combined/mosaic_hi"
res_df = pd.read_csv("zone1_hi_crossmatch_summary.csv")
top8 = res_df[res_df["Status"].isin(["DETECTION", "CANDIDATE"])].head(8)

fig, axes = plt.subplots(4, 2, figsize=(13, 14), dpi=150)
axes = axes.flatten()
f_rest = 1.420405752
c_kms = 299792.458

for i, (_, row) in enumerate(top8.iterrows()):
    ax = axes[i]
    spec_file = f"{hi_dir}/spectra_z1_csv/{row['Galaxy'].replace(' ', '_')}.csv"
    sp = pd.read_csv(spec_file)
    vel = c_kms * (f_rest / sp["freq_GHz"] - 1.0)
    flux = sp["flux_mJy"]
    rms = np.nanstd(flux)

    ax.step(vel, flux, where="mid", color="navy", lw=1.3, label="ATCA H I spectrum")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axhline(3 * rms, color="orange", ls="--", lw=0.9, label=f"3σ ({3*rms:.1f} mJy)")
    ax.axvline(row["cz_opt"], color="crimson", ls="--", lw=1.3, label=f"6dFGS cz = {int(row['cz_opt'])} km/s")
    ax.set_xlim(row["cz_opt"] - 1200, row["cz_opt"] + 1200)
    ax.set_title(f"{row['Galaxy']} [{row['Status']}, SNR={row['SNR']}]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Optical Velocity cz (km/s)", fontsize=9)
    ax.set_ylabel("Peak Flux (mJy/beam)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")

plt.tight_layout()
plt.savefig("zone1_hi_detections_spectra.png")
print("\nSUCCESS! Saved 8-panel H I spectra plot to: zone1_hi_detections_spectra.png")
