from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

v = Vizier(columns=["*"], row_limit=-1,
           vizier_server="vizier.cfa.harvard.edu")

# Download all tables in J/AJ/131/1280 directly without a coordinate cone search
res = v.get_catalogs("J/AJ/131/1280")
print("tables:", len(res))
for t in res:
    print(t.meta.get("name"), len(t), "rows")
    print(t.colnames)
    print(t[:5])

import numpy as np
from astropy.table import vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Combine table2 (main clusters) and table3 (supplementary clusters)
t2 = res[1]  # J/AJ/131/1280/table2
t3 = res[2]  # J/AJ/131/1280/table3

# Standardize velocity column name
t2_clean = t2['Cluster', 'RAJ2000', 'DEJ2000', '<cz>'].copy()
t2_clean.rename_column('<cz>', 'cz')

t3_clean = t3['Cluster', 'RAJ2000', 'DEJ2000', 'cz'].copy()
all_clusters = vstack([t2_clean, t3_clean])

cluster_coords = SkyCoord(all_clusters['RAJ2000'], all_clusters['DEJ2000'], unit=(u.hourangle, u.deg))
cluster_cz = np.array(all_clusters['cz'], dtype=float)

# 2. Check distances from your two ATCA galaxies to the nearest HRS clusters
sources = {
    "ATCA_18_0337-5345": (SkyCoord("03h37m06.66s", "-53d45m12.4s"), 0.075542, 22647),
    "ATCA_20_0335-5345": (SkyCoord("03h35m55.83s", "-53d45m12.2s"), 0.059641, 17880),
}

for s_name, (s_coord, z, s_cz) in sources.items():
    d_mpc = (299792.458 * z) / 70.0
    seps_deg = s_coord.separation(cluster_coords).deg
    proj_mpc = d_mpc * np.deg2rad(seps_deg)
    dv = s_cz - cluster_cz

    print(f"\n=== {s_name} (z={z:.6f}, cz={s_cz} km/s, D={d_mpc:.1f} Mpc) ===")
    # Sort by angular proximity and show top 5 closest clusters
    order = np.argsort(seps_deg)[:5]
    for idx in order:
        print(f"  {all_clusters['Cluster'][idx]:10s} | sep = {seps_deg[idx]:.2f} deg "
              f"({proj_mpc[idx]:.2f} Mpc) | cluster cz = {cluster_cz[idx]:.0f} km/s "
              f"(dv = {dv[idx]:+6.0f} km/s)")
