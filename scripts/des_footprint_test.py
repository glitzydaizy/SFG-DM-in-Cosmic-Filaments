from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

v = Vizier(columns=["*"], row_limit=5,
           vizier_server="vizier.cfa.harvard.edu")

tests = {
    "ATCA_18 field":      ("03h37m06.66s", "-53d45m12.4s"),
    "Dec -45":            ("03h37m06.66s", "-45d00m00s"),
    "Dec -60":            ("03h37m06.66s", "-60d00m00s"),
    "SN-X3 deep field":   ("03h30m35s", "-28d06m00s"),
}
for name, (ra, dec) in tests.items():
    pos = SkyCoord(ra, dec)
    t = v.query_region(pos, radius=300*u.arcsec, catalog="II/371/des_dr2")
    print(f"{name:18s} DES rows={len(t[0]) if len(t) else 0}")