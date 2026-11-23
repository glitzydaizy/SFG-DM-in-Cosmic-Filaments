import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.table import vstack, unique
import astropy.units as u

# Load final SFR table
df = pd.read_csv("results/d1213_sfg_sfr_v2.csv")

# Fetch HRS clusters from VizieR (Fleenor et al. 2006)
v = Vizier(columns=["*"], row_limit=-1, vizier_server="vizier.cfa.harvard.edu")
res = v.get_catalogs("J/AJ/131/1280")

t2 = res[1][['Cluster', 'RAJ2000', 'DEJ2000', '<cz>']].copy()
t2.rename_column('<cz>', 'cz')
t3 = res[2][['Cluster', 'RAJ2000', 'DEJ2000', 'cz']].copy()
clusters = unique(vstack([t2, t3]), keys='Cluster')

c_coords = SkyCoord(clusters['RAJ2000'], clusters['DEJ2000'], unit=(u.hourangle, u.deg))
c_ra = c_coords.ra.deg
c_dec = c_coords.dec.deg
c_cz = np.array(clusters['cz'], dtype=float)
c_names = np.array(clusters['Cluster'], dtype=str)

# Coordinates of two ATCA galaxies
src_coords = {
    "ATCA_18_0337-5345": SkyCoord("03h37m06.66s", "-53d45m12.4s"),
    "ATCA_20_0335-5345": SkyCoord("03h35m55.83s", "-53d45m12.2s"),
}

# =====================================================================
# CREATE TWO-PANEL THESIS FIGURE
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=250)
plt.subplots_adjust(wspace=0.28)

# --- PANEL A: Sky Map of the A3158-A3125-A3128 Filament Bridge ---
cen = SkyCoord("03h36m30s", "-53d45m12s")
local_mask = cen.separation(c_coords).deg < 3.2

# Draw dashed filament guide lines connecting A3158, APMCC 421, A3125, A3128
bridge_names = ["ACO 3158", "APMCC 421", "ACO 3125", "ACO 3128"]
bridge_pts = {}
for bname in bridge_names:
    m = (c_names == bname)
    if np.any(m):
        bridge_pts[bname] = (float(c_ra[m][0]), float(c_dec[m][0]))

for pair in [("ACO 3158", "APMCC 421"), ("APMCC 421", "ACO 3125"), ("APMCC 421", "ACO 3128")]:
    if pair[0] in bridge_pts and pair[1] in bridge_pts:
        x_vals = [bridge_pts[pair[0]][0], bridge_pts[pair[1]][0]]
        y_vals = [bridge_pts[pair[0]][1], bridge_pts[pair[1]][1]]
        ax1.plot(x_vals, y_vals, ls="--", color="0.65", lw=1.3, zorder=1)

# Plot local HRS clusters/groups
ax1.scatter(c_ra[local_mask], c_dec[local_mask], s=140, facecolors="none",
            edgecolors="black", linewidths=1.5, zorder=3, label="HRS Clusters / Groups")

for ra_val, dec_val, name, cz_val in zip(c_ra[local_mask], c_dec[local_mask],
                                         c_names[local_mask], c_cz[local_mask]):
    offset_y = 0.16 if name != "APMCC 421" else -0.32
    ax1.text(ra_val, dec_val + offset_y, f"{name}\n({cz_val:.0f} km/s)",
             fontsize=8, ha="center", va="center", color="0.2")

# Plot two ATCA sources
colors = {"ATCA_18_0337-5345": "C0", "ATCA_20_0335-5345": "C1"}
labels = {
    "ATCA_18_0337-5345": "ATCA_18 (z=0.0755, Outlier)",
    "ATCA_20_0335-5345": "ATCA_20 (z=0.0596, Filament)"
}
for s_id, sc in src_coords.items():
    ax1.scatter(sc.ra.deg, sc.dec.deg, marker="*", s=260, color=colors[s_id],
                edgecolors="black", linewidths=0.8, zorder=5, label=labels[s_id])

ax1.invert_xaxis()  # Astronomical convention: East to the left
ax1.set_xlim(56.6, 50.6)
ax1.set_ylim(-55.0, -51.7)
ax1.set_xlabel("Right Ascension J2000 (deg)", fontsize=10)
ax1.set_ylabel("Declination J2000 (deg)", fontsize=10)
ax1.set_title("(a) Sky Position in the HRS Filament Bridge", fontsize=11, fontweight="bold")
ax1.legend(loc="lower left", fontsize=8, frameon=True)
ax1.grid(True, linestyle=":", alpha=0.5)

# --- PANEL B: SFR Comparison & Error Budget ---
x = np.arange(len(df))
width = 0.34

# Murphy11 bars with propagated flux uncertainty
ax2.bar(x - width/2, df["SFR_Murphy11"], width, yerr=df["e_SFR_flux"],
        capsize=5, color=["C0", "C1"], edgecolor="black", alpha=0.88,
        label="Murphy et al. (2011) ± Flux Err")

# Overlay systematic spectral index range (alpha = 0.5 to 1.0) as thin brackets
yerr_low = np.abs(df["dSFR_alpha0.5"])
yerr_high = np.abs(df["dSFR_alpha1.0"])
ax2.errorbar(x - width/2 - 0.06, df["SFR_Murphy11"], yerr=[yerr_low, yerr_high],
             fmt="none", ecolor="darkred", elinewidth=1.5, capsize=4,
             label="Spectral Index Range (α = 0.5–1.0)")

# Bell03 comparison bars
ax2.bar(x + width/2, df["SFR_Bell03"], width, color=["C0", "C1"],
        edgecolor="black", hatch="//", alpha=0.45,
        label="Bell (2003) Calibration")

# Annotate exact values cleanly above the highest error bar cap
for idx, row in df.iterrows():
    top_err = max(row["e_SFR_flux"], row["dSFR_alpha1.0"])
    ax2.text(idx - width/2, row["SFR_Murphy11"] + top_err + 1.8,
             f"{row['SFR_Murphy11']:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(idx + width/2, row["SFR_Bell03"] + 1.8,
             f"{row['SFR_Bell03']:.1f}", ha="center", fontsize=9)

ax2.set_xticks(x)
ax2.set_xticklabels(["ATCA_18_0337-5345\n(z = 0.0755 | Outlier)",
                     "ATCA_20_0335-5345\n(z = 0.0596 | HRS Filament)"], fontsize=9.5)
ax2.set_ylabel("Star-Formation Rate, SFR (M☉ / yr)", fontsize=10)
ax2.set_ylim(0, 76)
ax2.set_title("(b) SFR Comparison & Uncertainty Budget", fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", fontsize=8, frameon=True)
ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

plt.savefig("results/thesis_summary_plots.png", dpi=250, bbox_inches="tight")
print("Saved results/thesis_summary_plots.png successfully!")
