import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.table import vstack, unique
import astropy.units as u

C = 299792.458
v = Vizier(columns=["*"], row_limit=-1,
           vizier_server="vizier.cfa.harvard.edu")
cen = SkyCoord("03h36m30s", "-53d45m12s")

tabs = []
for dra in (-6, -3, 0, 3, 6):
    for ddec in (-6, -3, 0, 3, 6):
        c = SkyCoord(cen.ra + (dra / np.cos(cen.dec.rad)) * u.deg,
                     cen.dec + ddec * u.deg)
        r = v.query_region(c, radius=3*u.deg, catalog="J/AJ/130/957")
        if len(r):
            tabs.append(r[0])
if not tabs:
    raise SystemExit("no tables returned from any pointing")
t = unique(vstack(tabs), keys="HRS")
gal = SkyCoord(t["RAJ2000"], t["DEJ2000"], unit=(u.hourangle, u.deg))
hrv = np.array(t["HRV"], dtype=float)
print(len(t), "galaxies in catalogue")

origin = SkyCoord("03h22m00s", "-51d19m00s")  # near the published HRS centre (A3120)
pa = np.deg2rad(-80.0)

def S(c):
    east = (c.ra - origin.ra).wrap_at(180*u.deg).deg * np.cos(origin.dec.rad)
    north = (c.dec - origin.dec).deg
    return east*np.sin(pa) + north*np.cos(pa)   # degrees along the axis

s_gal = S(gal)
core = (hrv > 17700) & (hrv < 22700)
slope, icpt = np.polyfit(s_gal[core], hrv[core], 1)
resid = hrv[core] - (slope*s_gal[core] + icpt)
print(f"core galaxies: {core.sum()}")
print(f"trend: {slope:.0f} km/s per degree, residual scatter {resid.std():.0f} km/s")

sources = {"ATCA_18": ("03h37m06.66s", "-53d45m12.4s", 22647),
           "ATCA_20": ("03h35m55.83s", "-53d45m12.2s", 17880)}
for name, (ra, dec, cz) in sources.items():
    s = S(SkyCoord(ra, dec))
    expected = slope*s + icpt
    print(f"{name}: S={s:+.2f} deg  cz={cz}  trend={expected:.0f}  "
          f"residual={cz-expected:+.0f} km/s")

plt.scatter(s_gal, hrv, s=6, c="0.75", label="all catalogue galaxies")
plt.scatter(s_gal[core], hrv[core], s=8, c="0.3", label="17,700-22,700 km/s")
for name, (ra, dec, cz) in sources.items():
    plt.scatter(S(SkyCoord(ra, dec)), cz, marker="*", s=200, label=name)
xs = np.linspace(s_gal.min(), s_gal.max(), 10)
plt.plot(xs, slope*xs + icpt, "k--", lw=1)
plt.xlabel("S along PA = -80 deg axis (deg)"); plt.ylabel("cz (km/s)")
plt.legend(fontsize=7)
plt.ylim(15000, 25500)
plt.savefig("results/hrs_axis_check.png", dpi=200, bbox_inches="tight")
print("wrote results/hrs_axis_check.png")
print()
print("residual percentiles (5/16/50/84/95):",
      np.percentile(resid, [5, 16, 50, 84, 95]).round(0))
for name, (ra, dec, cz) in sources.items():
    s = S(SkyCoord(ra, dec))
    r0 = cz - (slope*s + icpt)
    n = np.sum(abs(resid - r0) < 500)
    print(f"{name}: residual {r0:+.0f} km/s; {n} of {len(resid)} core galaxies "
          f"within +/-500 km/s of that residual ({100*n/len(resid):.0f}%)")
plt.figure()
plt.hist(resid, bins=np.arange(-3500, 3600, 250), color="0.6")
for name, (ra, dec, cz) in sources.items():
    s = S(SkyCoord(ra, dec))
    plt.axvline(cz - (slope*s + icpt), ls="--", label=name,
                color="C0" if name == "ATCA_18" else "C1")
plt.xlabel("residual from trend (km/s)"); plt.ylabel("N"); plt.legend()
plt.savefig("results/hrs_residual_hist.png", dpi=200, bbox_inches="tight")
pos18 = SkyCoord("03h37m06.66s", "-53d45m12.4s")
r18 = 22647 - (slope*S(pos18) + icpt)
print()
print("core galaxies within +/-500 km/s of ATCA_18's residual:")
for k, rr in zip(np.where(core)[0], resid):
    if abs(rr - r18) < 500:
        print(t["HRS"][k], f"cz={hrv[k]:.0f}",
              f"sep={pos18.separation(gal[k]).arcmin:.0f} arcmin")