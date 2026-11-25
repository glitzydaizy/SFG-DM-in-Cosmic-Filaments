import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your already-saved Zone 1 filament sample (zero network calls!)
orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
cat = pd.read_csv("zone1_4blk_hi_catalog.csv")
kin = pd.read_csv("zone1_starforming_kinematics.csv")[["Galaxy", "logMstar"]]

# Main Horologium-Reticulum filament sheet in Zone 1 (17,352 - 19,379 km/s, N = 15 galaxies)
main_sheet = orig[(orig["cz"] >= 17000) & (orig["cz"] <= 20000)].copy()
print(f"Tracing main filament spine using {len(main_sheet)} galaxies in the 17,350–19,380 km/s sheet.")

ra0, dec0 = 54.125, -53.3333
cos_dec0 = np.cos(np.radians(dec0))
x_s = (main_sheet["ra"].to_numpy() - ra0) * cos_dec0
y_s = main_sheet["dec"].to_numpy() - dec0

pts = np.column_stack([x_s, y_s])
cx0, cy0 = pts.mean(axis=0).ravel()[:2]
_, _, vt = np.linalg.svd(pts - np.array([cx0, cy0]))
vx, vy = vt.ravel()[:2]
norm = np.hypot(vx, vy)
sx, sy = vx / norm, vy / norm
px, py = -sy, sx

df = cat.merge(orig[["name", "ra", "dec", "z"]], left_on="Galaxy", right_on="name", how="left")
df = df.merge(kin, on="Galaxy", how="left")
df["logMstar"] = df["logMstar"].fillna(10.75)

c_kms = 299792.458
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

# --- PLOT 1: Cosmic Web Filament Spine Map + ATCA Sample ---
fig, ax = plt.subplots(figsize=(8.5, 7), dpi=150)
ax.scatter(orig["ra"], orig["dec"], c="lightgray", s=45, edgecolor="gray", label="All 6dFGS Zone 1 Filament Galaxies (N=25)")

t_line = np.linspace(-0.85, 0.85, 100)
spine_ra = ra0 + (cx0 + t_line * sx) / cos_dec0
spine_dec = dec0 + (cy0 + t_line * sy)
ax.plot(spine_ra, spine_dec, color="purple", ls="--", lw=2.3, label="Fitted Cosmic Filament Spine (PCA Ridge)")

theta = np.linspace(0, 2*np.pi, 200)
ax.plot(ra0 + (0.75 * np.cos(theta)) / cos_dec0, dec0 + 0.75 * np.sin(theta),
        color="black", ls=":", lw=1.4, label="ATCA Zone 1 Search Footprint (45' radius)")

sf = df[df["IR_Class"] == "Star-Forming"]
pas = df[df["IR_Class"] == "Passive/Quenched"]
unk = df[df["IR_Class"] == "Unknown"]

ax.scatter(sf["ra"], sf["dec"], c="dodgerblue", s=100, edgecolor="black", zorder=5, label="In-Field Star-Forming (W1-W3 > 1.5, N=9)")
ax.scatter(pas["ra"], pas["dec"], c="crimson", marker="^", s=100, edgecolor="black", zorder=5, label="In-Field Passive/Quenched (W1-W3 ≤ 1.5, N=7)")
if len(unk) > 0:
    ax.scatter(unk["ra"], unk["dec"], c="gold", marker="s", s=80, edgecolor="black", zorder=5, label="In-Field Unclassified (N=3)")

ax.invert_xaxis()
ax.set_xlabel("Right Ascension J2000 (deg)", fontsize=11)
ax.set_ylabel("Declination J2000 (deg)", fontsize=11)
ax.set_title("Cosmic Web Filament Spine & ATCA Zone 1 Sample Distribution", fontsize=11.5, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig("zone1_filament_map.png")
print("Saved: zone1_filament_map.png")

# --- PLOT 2: Environmental Trends vs. Filament Spine Distance ---
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
print("Saved: zone1_filament_trends.png")

# --- PLOT 3: Full Dark Matter Halo Density & Enclosed Mass Profiles (NFW vs. Burkert vs. Pseudo-Isothermal) ---
r_opt = 15.0  # kpc
m_dm_opt = 2.70e11  # M_sun enclosed DM mass at R_opt
r_grid = np.logspace(np.log10(0.5), np.log10(150.0), 200)  # 0.5 to 150 kpc

# 1. NFW Halo (Cuspy: rho ~ r^-1 in center, r^-3 outer)
r_s = 18.0
x_opt = r_opt / r_s
rho_s = m_dm_opt / (4.0 * np.pi * (r_s**3) * (np.log(1.0 + x_opt) - x_opt / (1.0 + x_opt)))
rho_nfw = rho_s / ((r_grid / r_s) * ((1.0 + r_grid / r_s)**2))
m_nfw = 4.0 * np.pi * rho_s * (r_s**3) * (np.log(1.0 + r_grid / r_s) - (r_grid / r_s) / (1.0 + r_grid / r_s))

# 2. Burkert Halo (Cored: constant rho_0 in center, r^-3 outer)
r_0 = 10.0
xb_opt = r_opt / r_0
rho_0 = m_dm_opt / (np.pi * (r_0**3) * (np.log(((1.0 + xb_opt)**2) * (1.0 + xb_opt**2)) - 2.0 * np.arctan(xb_opt)))
rho_burk = (rho_0 * (r_0**3)) / ((r_grid + r_0) * (r_grid**2 + r_0**2))
m_burk = np.pi * rho_0 * (r_0**3) * (np.log(((1.0 + r_grid / r_0)**2) * (1.0 + (r_grid / r_0)**2)) - 2.0 * np.arctan(r_grid / r_0))

# 3. Pseudo-Isothermal Halo (Cored: rho ~ (1 + (r/r_c)^2)^-1)
r_c = 6.0
xc_opt = r_opt / r_c
rho_iso0 = m_dm_opt / (4.0 * np.pi * (r_c**3) * (xc_opt - np.arctan(xc_opt)))
rho_iso = rho_iso0 / (1.0 + (r_grid / r_c)**2)
m_iso = 4.0 * np.pi * rho_iso0 * (r_c**3) * ((r_grid / r_c) - np.arctan(r_grid / r_c))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

ax1.loglog(r_grid, rho_nfw, color="crimson", lw=2.2, label="NFW Halo (Cuspy: ρ ∝ r⁻¹)")
ax1.loglog(r_grid, rho_burk, color="darkorange", ls="--", lw=2.2, label="Burkert Halo (Cored)")
ax1.loglog(r_grid, rho_iso, color="teal", ls="-.", lw=2.0, label="Pseudo-Isothermal Halo (Cored)")
ax1.axvline(r_opt, color="black", ls=":", lw=1.3, label="Optical Disk Radius R_opt (15 kpc)")
ax1.set_xlabel("Galactocentric Radius R (kpc)", fontsize=10)
ax1.set_ylabel("Dark Matter Density ρ_DM(R) (M☉ / kpc³)", fontsize=10)
ax1.set_title("A. Radial Dark Matter Halo Density Profiles", fontsize=10.5, fontweight="bold")
ax1.legend(fontsize=8.5)

ax2.loglog(r_grid, m_nfw, color="crimson", lw=2.2, label="NFW Enclosed Mass M_DM(<R)")
ax2.loglog(r_grid, m_burk, color="darkorange", ls="--", lw=2.2, label="Burkert Enclosed Mass M_DM(<R)")
ax2.loglog(r_grid, m_iso, color="teal", ls="-.", lw=2.0, label="Pseudo-Isothermal M_DM(<R)")
ax2.plot([r_opt], [m_dm_opt], "ko", markersize=8, zorder=5, label=f"Measured M_DM(R_opt) = {m_dm_opt:.2e} M☉")
ax2.axhline(6.89e10, color="royalblue", ls=":", lw=1.6, label="Total Baryonic Mass M_bar = 6.89×10¹⁰ M☉")
ax2.set_xlabel("Galactocentric Radius R (kpc)", fontsize=10)
ax2.set_ylabel("Enclosed Mass M(<R) (M☉)", fontsize=10)
ax2.set_title("B. Cumulative Dark Matter vs. Baryonic Mass Profile", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8.5, loc="lower right")

plt.tight_layout()
plt.savefig("zone1_dm_halo_density_profiles.png")
print("Saved: zone1_dm_halo_density_profiles.png")
print(f"NFW Virial Mass estimate M_200 (~150 kpc): {m_nfw[-1]:.2e} M_sun (log M_200 = {np.log10(m_nfw[-1]):.2f})")
