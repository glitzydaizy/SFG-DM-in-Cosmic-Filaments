import time
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

v = Vizier(columns=["*"], row_limit=-1,
           vizier_server="vizier.cfa.harvard.edu")
hosts = {"ATCA_20 host": SkyCoord("03h35m55.94s", "-53d45m10.8s"),
         "ATCA_18 host": SkyCoord("03h37m06.65s", "-53d45m12.5s")}

for name, pos in hosts.items():
    print("=" * 60)
    print(name)
    for attempt in range(3):
        res = v.query_region(pos, radius=30*u.arcsec,
                             catalog="J/AJ/131/1280/table1")
        if len(res):
            res[0].pprint_all()
            break
        time.sleep(3)
    else:
        print("no rows after 3 tries")
        