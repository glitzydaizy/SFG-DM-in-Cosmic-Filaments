from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

v = Vizier(columns=["*"], row_limit=-1,
           vizier_server="vizier.cfa.harvard.edu")
pos = SkyCoord("03h36m30s", "-53d45m12s")   # between my two sources
res = v.query_region(pos, radius=3*u.deg, catalog="J/AJ/130/957")

print("tables returned:", len(res))
for t in res:
    print(t.meta.get("name"), len(t), "rows")
    print(t.colnames)
    print(t[:5])