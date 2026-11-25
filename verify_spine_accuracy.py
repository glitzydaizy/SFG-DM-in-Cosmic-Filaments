import requests
from io import BytesIO
import numpy as np
import pandas as pd
from astropy.table import Table

r = requests.get(
    "https://nasa.gov",
    params={"table": "sixdfgs", "RA": 54.125, "DEC": -53.3333, "SR": 0.75},
    timeout=60,
)
t = Table.read(BytesIO(r.content), format="votable").to_pandas()
sheet = t[(t["radial_velocity"] >= 17000) & (t["radial_velocity"] <= 20000)].copy()

ra0, dec0 = 54.125, -53.3333
cos_d = np.cos(np.radians(dec0))
x = (sheet["ra"].to_numpy() - ra0) * cos_d
y = sheet["dec"].to_numpy() - dec0

# 1. Unweighted PCA (all 15 galaxies)
pts = np.column_stack([x, y])
c_unw = pts.mean(axis=0)
_, _, vt_unw = np.linalg.svd(pts - c_unw)
ang_unw = np.degrees(np.arctan2(vt_unw[0, 1], vt_unw[0, 0]))

# 2. Local-density weighted PCA (weights galaxies in denser ridge more strongly, reducing outlier pull)
dists = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
rho_local = np.sum(np.exp(-0.5 * (dists / 0.25) ** 2), axis=1)
w = rho_local / np.sum(rho_local)
c_w = np.sum(pts * w[:, None], axis=0)
_, _, vt_w = np.linalg.svd((pts - c_w) * np.sqrt(w[:, None]))
ang_w = np.degrees(np.arctan2(vt_w[0, 1], vt_w[0, 0]))

shift_deg = np.hypot(*(c_unw - c_w))
d_a = ((299792.458 / 70.0) * 0.061) / (1.061 ** 2)
shift_mpc = d_a * np.radians(shift_deg)

print(f"Unweighted 15-galaxy PCA : Center Dec = {dec0 + c_unw[1]:.3f}°, Orientation PA = {ang_unw:.1f}°")
print(f"Density-Weighted Ridge   : Center Dec = {dec0 + c_w[1]:.3f}°, Orientation PA = {ang_w:.1f}°")
print(f"Spine Offset Difference  : {shift_deg:.3f}° ({shift_mpc:.2f} Mpc) | Angle Diff = {abs(ang_unw - ang_w):.1f}°")
