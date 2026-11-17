from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

v = Vizier(columns=["*"], row_limit=5,
           vizier_server="vizier.cfa.harvard.edu")
c = SkyCoord("03h37m04.692s -53d01m21.42s")

for cat in ["II/371/des_dr2", "J/ApJS/225/5/wisc"]:
    res = v.query_region(c, radius=3*u.arcmin, catalog=cat)
    print(cat, ":", len(res), "table(s) returned")
    if len(res):
        res[0].pprint(max_width=-1, max_lines=5)
