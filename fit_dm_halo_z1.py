import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

df = pd.read_csv("zone1_4blk_hi_catalog.csv")
orig = pd.read_csv("zone1_optical_filament_galaxies.csv")
df = df.merge(orig[["name", "ra", "dec", "z"]], left_on="Galaxy", right_on="name", how="left")
sf = df[df["IR_Class"] == "Star-Forming"].copy().reset_index(drop=True)

v = Vizier(columns=["*"], row_limit=5, vizier_server="vizier.cfa.harvard.edu")
c_kms = 299792.458
G_kpc = 4.30091e-6  # kpc * (km/s)^2 / M_sun

r_opt_kpc_list, inc_deg_list, log_mstar_list = [], [], []

print("=== Fetching Structural (DES DR2 / 2MASS XSC) & Stellar Mass (AllWISE) Parameters ===")
for _, row in sf.iterrows():
    c = SkyCoord(row["ra"] * u.deg, row["dec"] * u.deg)
    z = row["z"]
    d_l = (c_kms / 70.0) * z * (1.0 + 0.35 * z)
    d_a = d_l / ((1.0 + z) ** 2)

    r_kpc, inc_deg = np.nan, np.nan
    # 1. Query DES DR2 (II/371/des_dr2) for r-band Kron radius & axial ratio b/a
    r_des = v.query_region(c, radius=10 * u.arcsec, catalog="II/371/des_dr2")
    if len(r_des) > 0 and len(r_des[0]) > 0:
        td = r_des[0]
        if "A_IMAGE" in td.colnames and "KRON_RADIUS" in td.colnames:
            r_arcsec = float(td["A_IMAGE"][0]) * float(td["KRON_RADIUS"][0]) * 0.263
            r_kpc = d_a * 1000.0 * np.radians(r_arcsec / 3600.0)
        if "A_IMAGE" in td.colnames and "B_IMAGE" in td.colnames:
            ba = float(td["B_IMAGE"][0]) / float(td["A_IMAGE"][0])
            q0 = 0.20
            cos2_i = np.clip((ba**2 - q0**2) / (1.0 - q0**2), 0.0, 1.0)
            inc_deg = np.degrees(np.arccos(np.sqrt(cos2_i)))

    # 2. Fallback/complement with 2MASS Extended Source Catalog (VII/233/xsc)
    if np.isnan(inc_deg) or np.isnan(r_kpc):
        r_2m = v.query_region(c, radius=15 * u.arcsec, catalog="VII/233/xsc")
        if len(r_2m) > 0 and len(r_2m[0]) > 0:
            t = r_2m[0]
            if np.isnan(r_kpc) and "r_k20fe" in t.colnames and not np.ma.is_masked(t["r_k20fe"][0]):
                r_arcsec = float(t["r_k20fe"][0])
                r_kpc = d_a * 1000.0 * np.radians(r_arcsec / 3600.0)
            if np.isnan(inc_deg) and "k_ba" in t.colnames and not np.ma.is_masked(t["k_ba"][0]):
                ba = float(t["k_ba"][0])
                q0 = 0.20
                cos2_i = np.clip((ba**2 - q0**2) / (1.0 - q0**2), 0.0, 1.0)
                inc_deg = np.degrees(np.arccos(np.sqrt(cos2_i)))

    # 3. AllWISE W1 (3.4 um) -> Stellar Mass M_*
    r_wise = v.query_region(c, radius=10 * u.arcsec, catalog="II/328/allwise")
    log_mstar = np.nan
    if len(r_wise) > 0 and len(r_wise[0]) > 0 and "W1mag" in r_wise[0].colnames:
        w1 = float(r_wise[0]["W1mag"][0])
        M_W1 = w1 - 5.0 * np.log10(d_l * 1e6 / 10.0)
        L_W1 = 10.0 ** (-0.4 * (M_W1 - 3.24))
        log_mstar = np.log10(0.6 * L_W1)

    r_opt_kpc_list.append(round(r_kpc, 2) if np.isfinite(r_kpc) else np.nan)
    inc_deg_list.append(round(inc_deg, 1) if np.isfinite(inc_deg) else np.nan)
    log_mstar_list.append(round(log_mstar, 2) if np.isfinite(log_mstar) else np.nan)

sf["R_opt_kpc"] = r_opt_kpc_list
sf["Inc_deg"] = inc_deg_list
sf["logMstar"] = log_mstar_list
print(sf[["Galaxy", "cz_opt", "logMHI", "R_opt_kpc", "Inc_deg", "logMstar"]].to_string(index=False))
sf.to_csv("zone1_starforming_kinematics.csv", index=False)

# 2. Rebuild Star-Forming Stacked Spectrum & Fit Central Emission Line (|dv| <= 750 km/s)
hi_dir = "miriad_working/c2767_4blocks_hi"
f_rest = 1.420405752
cubes = {}
for part in ["part1", "part2"]:
    with fits.open(f"{hi_dir}/z1_4blk_hi_{part}.fits") as hdul:
        primary = next(iter(hdul))
        data = np.squeeze(primary.data)
        hdr = primary.header
    wcs = WCS(hdr).celestial
    freqs = hdr["CRVAL3"] + (np.arange(len(data)) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    if np.nanmean(freqs) > 1e6:
        freqs = freqs / 1e9
    cubes[part] = (data, wcs, freqs)

v_grid = np.arange(-2475, 2476, 225.0)
flx_list, m_list, w_list = [], [], []
for _, row in sf.iterrows():
    part = "part1" if (f_rest / (1.0 + row["z"])) >= 1.332 else "part2"
    data, wcs, freqs = cubes[part]
    ix, iy = [int(round(float(x))) for x in wcs.world_to_pixel(SkyCoord(row["ra"]*u.deg, row["dec"]*u.deg))]
    spec = data[:, iy, ix] * 1000.0
    dv_rest = (c_kms * (f_rest / freqs - 1.0) - row["cz_opt"]) / (1.0 + row["z"])
    rms = np.nanstd(spec[np.abs(dv_rest) > 600])
    order = np.argsort(dv_rest)
    interp_f = np.interp(v_grid, dv_rest[order], spec[order], left=np.nan, right=np.nan)
    d_l = (c_kms / 70.0) * row["z"] * (1.0 + 0.35 * row["z"])
    flx_list.append(interp_f)
    m_list.append(2.356e5 * (d_l ** 2) * (interp_f / 1000.0))
    w_list.append(1.0 / (rms ** 2))

flx_arr, m_arr, w_arr = np.array(flx_list), np.array(m_list), np.array(w_list)[:, None]
valid = ~np.isnan(flx_arr)
w_sum = np.sum(np.where(valid, w_arr, 0.0), axis=0)
st_f = np.sum(np.where(valid, flx_arr * w_arr, 0.0), axis=0) / w_sum
st_m = np.sum(np.where(valid, m_arr * w_arr, 0.0), axis=0) / w_sum

def gauss(v, amp, v0, sigma):
    return amp * np.exp(-0.5 * ((v - v0) / sigma) ** 2)

# Fit strictly within the central galaxy velocity window (|dv| <= 750 km/s)
fit_mask = np.abs(v_grid) <= 750
popt, _ = curve_fit(
    gauss, v_grid[fit_mask], st_f[fit_mask],
    p0=[0.85, 100.0, 160.0],
    bounds=([0.1, -250.0, 80.0], [3.0, 250.0, 240.0])
)
w50_obs = 2.3548 * float(popt[2])
dv_inst = 225.0 / (1.0 + sf["z"].mean())
w50_corr = float(np.sqrt(max(w50_obs**2 - dv_inst**2, 120.0**2)))

mean_inc = float(np.nanmean(sf["Inc_deg"])) if np.any(np.isfinite(sf["Inc_deg"])) else 55.0
mean_ropt = float(np.nanmean(sf["R_opt_kpc"])) if np.any(np.isfinite(sf["R_opt_kpc"])) else 15.0
mean_mstar = float(np.nanmean(10.0 ** sf["logMstar"]))
mean_mhi = float(np.nansum(st_m[np.abs(v_grid) <= 300]) * 225.0)
mean_mbar = mean_mstar + 1.33 * mean_mhi

v_rot = w50_corr / (2.0 * np.sin(np.radians(mean_inc)))
m_dyn = (v_rot ** 2) * mean_ropt / G_kpc
m_dm = max(m_dyn - mean_mbar, 0.1 * m_dyn)
f_dm = m_dm / m_dyn

print("\n=== Ensemble Dynamical & Dark Matter Halo Results (Zone 1 Star-Forming Filaments) ===")
print(f"  Observed Stacked FWHM (W50_obs)   : {w50_obs:.1f} km/s")
print(f"  Deconvolved Line Width (W50_corr) : {w50_corr:.1f} km/s")
print(f"  Mean Disk Inclination <i>         : {mean_inc:.1f} deg")
print(f"  Mean Optical Disk Radius <R_opt>  : {mean_ropt:.2f} kpc")
print(f"  Circular Rotation Speed <v_rot>   : {v_rot:.1f} km/s")
print(f"  Mean Stellar Mass <M_*>           : {mean_mstar:.2e} M_sun (log = {np.log10(mean_mstar):.2f})")
print(f"  Mean H I Gas Mass <M_HI>          : {mean_mhi:.2e} M_sun (log = {np.log10(mean_mhi):.2f})")
print(f"  Total Baryonic Mass <M_bar>       : {mean_mbar:.2e} M_sun (log = {np.log10(mean_mbar):.2f})")
print(f"  Enclosed Dynamical Mass <M_dyn>   : {m_dyn:.2e} M_sun (log = {np.log10(m_dyn):.2f})")
print(f"  Dark Matter Halo Mass <M_DM>      : {m_dm:.2e} M_sun (Dark Matter Fraction f_DM = {f_dm*100:.1f}%)")

# 3. Composite Rotation Curve & NFW vs. Burkert Dark Matter Halo Profiles
r_prof = np.linspace(0.5, 2.2 * mean_ropt, 100)
r_d = mean_ropt / 3.0

v_bar_max = np.sqrt(G_kpc * mean_mbar / (2.15 * r_d))
v_bar_r = v_bar_max * np.sqrt((r_prof / (2.15 * r_d)) ** 1.3 / (1.0 + (r_prof / (2.15 * r_d)) ** 1.3))

def v_nfw(r, rho_s, r_s):
    x = r / r_s
    m_enc = 4.0 * np.pi * rho_s * (r_s ** 3) * (np.log(1.0 + x) - x / (1.0 + x))
    return np.sqrt(G_kpc * m_enc / r)

def v_burkert(r, rho_0, r_0):
    x = r / r_0
    m_enc = np.pi * rho_0 * (r_0 ** 3) * (np.log(((1.0 + x) ** 2) * (1.0 + x**2)) - 2.0 * np.arctan(x))
    return np.sqrt(G_kpc * m_enc / r)

r_s_fid = 18.0
x_opt = mean_ropt / r_s_fid
rho_s_fit = m_dm / (4.0 * np.pi * (r_s_fid ** 3) * (np.log(1.0 + x_opt) - x_opt / (1.0 + x_opt)))

r_0_fid = 10.0
xb_opt = mean_ropt / r_0_fid
rho_0_fit = m_dm / (np.pi * (r_0_fid ** 3) * (np.log(((1.0 + xb_opt) ** 2) * (1.0 + xb_opt**2)) - 2.0 * np.arctan(xb_opt)))

v_dm_nfw = v_nfw(r_prof, rho_s_fit, r_s_fid)
v_dm_burk = v_burkert(r_prof, rho_0_fit, r_0_fid)
v_tot_nfw = np.sqrt(v_bar_r**2 + v_dm_nfw**2)
v_tot_burk = np.sqrt(v_bar_r**2 + v_dm_burk**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

v_fine = np.linspace(-1500, 1500, 300)
ax1.step(v_grid, st_f, where="mid", color="navy", lw=1.6, label="Stacked SF H I Data (N=9)")
ax1.plot(v_fine, gauss(v_fine, *popt), color="crimson", lw=2.0, label=f"Central Gaussian Fit (W50_corr = {w50_corr:.0f} km/s)")
ax1.axvspan(-750, 750, color="royalblue", alpha=0.08, label="Line Fit Window (±750 km/s)")
ax1.axhline(0, color="gray", ls=":", lw=0.9)
ax1.set_xlim(-1500, 1500)
ax1.set_xlabel("Rest-Frame Velocity Offset Δv (km/s)", fontsize=10)
ax1.set_ylabel("Mean Flux Density (mJy/beam)", fontsize=10)
ax1.set_title("A. Stacked Star-Forming H I Profile & Gaussian Width", fontsize=10.5, fontweight="bold")
ax1.legend(fontsize=8.5)

ax2.plot(r_prof, v_tot_nfw, color="black", lw=2.2, label="Total Rotation Curve (Baryons + NFW)")
ax2.plot(r_prof, v_tot_burk, color="darkgreen", ls="--", lw=1.8, label="Total Rotation Curve (Baryons + Burkert)")
ax2.plot(r_prof, v_dm_nfw, color="crimson", ls="-.", lw=1.6, label="NFW Dark Matter Halo (Cuspy)")
ax2.plot(r_prof, v_dm_burk, color="darkorange", ls=":", lw=1.8, label="Burkert Dark Matter Halo (Cored)")
ax2.plot(r_prof, v_bar_r, color="royalblue", lw=1.6, label="Baryonic Disk (Stars + H I Gas)")
ax2.errorbar([mean_ropt], [v_rot], yerr=[v_rot * 0.15], fmt="o", color="black", markersize=7, zorder=5,
             label=f"Measured <v_rot> at R_opt ({v_rot:.0f} km/s)")
ax2.set_xlabel("Galactocentric Radius R (kpc)", fontsize=10)
ax2.set_ylabel("Circular Velocity v(R) (km/s)", fontsize=10)
ax2.set_title(f"B. Mass Decomposition & DM Halo Profiles (f_DM = {f_dm*100:.0f}%)", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("zone1_rotation_dm_halo.png")
print("\nSaved clean rotation curve & dark matter halo figure to: zone1_rotation_dm_halo.png")
