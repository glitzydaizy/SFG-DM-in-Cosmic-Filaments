import requests
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

print("Querying 2.0-deg diameter (SR=1.0 deg) 6dFGS field via HEASARC...")
r = requests.get(
    "https://nasa.gov",
    params={"table": "sixdfgs", "RA": 54.125, "DEC": -53.3333, "SR": 1.0},
    timeout=60,
)
t_wide = Table.read(BytesIO(r.content), format="votable").to_pandas()
wide_fil = t_wide[(t_wide["radial_velocity"] >= 16500) & (t_wide["radial_velocity"] <= 20000)].copy()
print(f"Found {len(wide_fil)} galaxies tracing the main Horologium-Reticulum filament sheet (16,500 - 20,000 km/s).")

ra0, dec0 = 54.125, -53.3333
cos_dec0 = np.cos(np.radians(dec0))
x_wide = (wide_fil["ra"].to_numpy() - ra0) * cos_dec0
y_wide = wide_fil["dec"].to_numpy() - dec0

pts = np.column_stack([x_wide, y_wide])
center_xy = pts.mean(axis=0)
_, _, vt = np.linalg.svd(pts - center_xy)
vx, vy = vt.ravel()[:2]
norm = np.hypot(vx, vy)
sx, sy = vx / norm, vy / norm
px, py = -sy, sx

cat = pd.read_csv("zone1_4blk_hi_catalog.csv")
orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
kin = pd.read_csv("zone1_starforming_kinematics.csv")[["Galaxy", "logMstar"]]
df = cat.merge(orig[["name", "ra", "dec", "z"]], left_on="Galaxy", right_on="name", how="left")
df = df.merge(kin, on="Galaxy", how="left")
df["logMstar"] = df["logMstar"].fillna(10.75)

c_kms = 299792.458
cx0, cy0 = center_xy.ravel()[:2]
d_fil_mpc = []
for _, row in df.iterrows():
    gx = (row["ra"] - ra0) * cos_dec0 - cx0
    gy = (row["dec"] - dec0) - cy0
    sep_deg = abs(gx * px + gy * py)
    z = row["z"]
    d_a = ((c_kms / 70.0) * z * (1.0 + 0.35 * z)) / ((1.0 + z) ** 2)
    d_fil_mpc.append(round(d_a * np.radians(sep_deg), 3))

df["D_fil_Mpc"] = d_fil_mpc
df["log_fHI_lim"] = df["logMHI_3sig_lim"] - df["logMstar"]
df.to_csv("zone1_filament_environment.csv", index=False)

# --- FIGURE 1: Cosmic Filament Map + ATCA Zone 1 Sample ---
fig, ax = plt.subplots(figsize=(8.5, 7), dpi=150)
ax.scatter(wide_fil["ra"], wide_fil["dec"], c="lightgray", s=40, edgecolor="gray", label="Wider 6dFGS Filament Sheet (16.5–20k km/s)")

t_line = np.linspace(-1.0, 1.0, 100)
spine_ra = ra0 + (cx0 + t_line * sx) / cos_dec0
spine_dec = dec0 + (cy0 + t_line * sy)
ax.plot(spine_ra, spine_dec, color="purple", ls="--", lw=2.2, label="Fitted Cosmic Filament Spine (PCA)")

theta = np.linspace(0, 2*np.pi, 200)
ax.plot(ra0 + (0.75 * np.cos(theta)) / cos_dec0, dec0 + 0.75 * np.sin(theta),
        color="black", ls=":", lw=1.4, label="ATCA Zone 1 Mosaic Footprint (45' radius)")

sf = df[df["IR_Class"] == "Star-Forming"]
pas = df[df["IR_Class"] == "Passive/Quenched"]
unk = df[df["IR_Class"] == "Unknown"]

ax.scatter(sf["ra"], sf["dec"], c="dodgerblue", s=95, edgecolor="black", zorder=5, label="ATCA Star-Forming (W1-W3 > 1.5)")
ax.scatter(pas["ra"], pas["dec"], c="crimson", marker="^", s=95, edgecolor="black", zorder=5, label="ATCA Passive/Quenched (W1-W3 ≤ 1.5)")
if len(unk) > 0:
    ax.scatter(unk["ra"], unk["dec"], c="gold", marker="s", s=75, edgecolor="black", zorder=5, label="ATCA Unclassified")

ax.invert_xaxis()
ax.set_xlabel("Right Ascension J2000 (deg)", fontsize=11)
ax.set_ylabel("Declination J2000 (deg)", fontsize=11)
ax.set_title("Cosmic Web Filament Structure & ATCA Zone 1 Sample Distribution", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig("zone1_filament_map.png")
print("Saved filament sky map to: zone1_filament_map.png")

# --- FIGURE 2: Environmental Pre-Processing Trends vs. Distance to Filament Spine ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

ax1.errorbar(sf["D_fil_Mpc"], sf["log_fHI_lim"], yerr=0.15, uplims=True, fmt="o",
             color="dodgerblue", markersize=7, label="Star-Forming (3σ Upper Limits)")
ax1.errorbar(pas["D_fil_Mpc"], pas["log_fHI_lim"], yerr=0.15, uplims=True, fmt="^",
             color="crimson", markersize=7, label="Passive/Quenched (3σ Upper Limits)")
ax1.axhline(np.log10(8.31e9 / 5.78e10), color="dodgerblue", ls="--", lw=1.6,
            label="SF Stacked Mean log(M_HI/M_*) = -0.84")
ax1.axhline(np.log10(3.69e9 / 5.62e10), color="crimson", ls=":", lw=1.6,
            label="Passive Stacked Limit log(M_HI/M_*) < -1.18")
ax1.set_xlabel("Projected Distance to Filament Spine D_fil (Mpc)", fontsize=10)
ax1.set_ylabel("log10(M_HI / M_*) [H I Gas Fraction]", fontsize=10)
ax1.set_title("A. H I Gas Fraction vs. Filament Spine Distance", fontsize=10.5, fontweight="bold")
ax1.legend(fontsize=8)

categories = ["All Filament\nGalaxies (N=19)", "Star-Forming\n(W1-W3 > 1.5, N=9)", "Passive/Quenched\n(W1-W3 ≤ 1.5, N=7)"]
masses = [6.42, 8.31, 3.69]
colors = ["slategray", "dodgerblue", "crimson"]
ax2.bar(categories, masses, color=colors, edgecolor="black", width=0.52)
ax2.annotate("Non-Detection\n(SNR = 0.61σ)", xy=(2, 3.85), ha="center", fontsize=8.5, color="crimson", fontweight="bold")
ax2.annotate("Detected Stack\n(SNR = 2.97σ)", xy=(1, 8.5), ha="center", fontsize=8.5, color="navy", fontweight="bold")
ax2.set_ylim(0, 10.2)
ax2.set_ylabel("Average H I Mass <M_HI> (× 10⁹ M☉)", fontsize=10)
ax2.set_title("B. Cold H I Gas Depletion Across Filament Populations", fontsize=10.5, fontweight="bold")

plt.tight_layout()
plt.savefig("zone1_filament_trends.png")
print("Saved filament environmental trends to: zone1_filament_trends.png")
