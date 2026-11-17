from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

v = Vizier(columns=["*"], row_limit=5)   # no mirror: default CDS server
c = SkyCoord("03h37m04.692s -53d01m21.42s")
print(v.query_region(c, radius=3*u.arcmin, catalog="II/371/des_dr2"))
