from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

v = Vizier(columns=["*"], row_limit=50,
           vizier_server="vizier.cfa.harvard.edu")

sources = {"source1": "03h37m04.692s -53d01m21.42s",
           "source2": "03h37m05.876s -52d58m20.33s"}

cols = ["AllWISE", "sep_arcsec", "W1mag", "W2mag", "W3mag", "e_W3mag",
        "W4mag", "e_W4mag", "W1-W2", "ccf", "qph"]

for name, pos in sources.items():
    c = SkyCoord(pos)
    res = v.query_region(c, radius=10*u.arcsec, catalog="II/328/allwise")
    if not len(res):
        print(name, ": no AllWISE match within 10 arcsec")
        continue
    t = res[0]
    t["sep_arcsec"] = c.separation(
        SkyCoord(t["RAJ2000"], t["DEJ2000"])).arcsec
    t.sort("sep_arcsec")
    t["W1-W2"] = t["W1mag"] - t["W2mag"]
    print("\n", name)
    t[cols].pprint(max_width=-1)

