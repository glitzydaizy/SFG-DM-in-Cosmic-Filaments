import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

C = 299792.458
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

v = Vizier(columns=["*"], row_limit=-1,
           vizier_server="vizier.cfa.harvard.edu")
cen = SkyCoord("03h36m30s", "-53d45m12s")
t = v.query_region(cen, radius=3*u.deg, catalog="J/AJ/130/957")[0]

gal = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.hourangle, u.deg))
hrv = np.array(t["HRV"], dtype=float)

print(f"{len(t)} galaxies, HRV {hrv.min():.0f}-{hrv.max():.0f} km/s")
print(f"mean {hrv.mean():.0f}, median {np.median(hrv):.0f}, std {hrv.std():.0f} km/s")
print("percentiles 5/16/50/84/95:",
      np.percentile(hrv, [5, 16, 50, 84, 95]).round(0))
print()

sources = {
    "ATCA_18": ("03h37m06.66s", "-53d45m12.4s", 0.075542),
    "ATCA_20": ("03h35m55.83s", "-53d45m12.2s", 0.059641),
}

for name, (ra, dec, z) in sources.items():
    pos = SkyCoord(ra, dec)
    cz = C * z
    dv = (hrv - cz) / (1 + z)            # rest-frame velocity offset
    sep = pos.separation(gal).arcmin
    da = cosmo.angular_diameter_distance(z).to(u.kpc).value
    proj = sep / 60 * np.pi / 180 * da   # projected kpc
    print(f"{name}: cz = {cz:.0f} km/s")
    print(f"  catalogue galaxies within +/-1000 km/s: {np.sum(abs(dv) < 1000)}")
    print(f"  within +/-3000 km/s: {np.sum(abs(dv) < 3000)}")
    print(f"  nearest 5 on the sky:")
    for i in np.argsort(sep)[:5]:
        print(f"    {t['HRS'][i]}  sep={sep[i]:5.1f}'  ({proj[i]/1000:.2f} Mpc)  "
              f"cz={hrv[i]:.0f}  dv={dv[i]:+.0f} km/s")
    print()

plt.hist(hrv, bins=np.arange(10000, 30000, 500), color="0.7")
for name, (_, _, z) in sources.items():
    plt.axvline(C * z, label=name, ls="--",
                color="C3" if name == "ATCA_18" else "C0")
plt.xlabel("cz (km/s)"); plt.ylabel("N"); plt.legend()
plt.savefig("results/hrs_redshift_check.png", dpi=200, bbox_inches="tight")
print("wrote results/hrs_redshift_check.png")

print()
print("ATCA_18: galaxies within +/-1000 km/s of its cz")
pos = SkyCoord("03h37m06.66s", "-53d45m12.4s")
cz = C * 0.075542
sep = pos.separation(gal).arcmin
for i in np.argsort(abs(hrv - cz)):
    if abs(hrv[i] - cz) > 1000:
        break
    print(t["HRS"][i], f"sep={sep[i]:.0f}'", f"cz={hrv[i]:.0f}")