from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

v = Vizier(columns=["*"], row_limit=20,
           vizier_server="vizier.cfa.harvard.edu")

sources = {
    "source1": "03h37m04.692s -53d01m21.42s",
    "source2": "03h37m05.876s -52d58m20.33s"
}

catalogs = {
    "6dFGS": "VII/259/6dfgs",
    "HRS_Fleenor2005": "J/AJ/130/957"
}

for name, pos in sources.items():
    c = SkyCoord(pos)
    print(f"\n=== {name} ({pos}) ===")
    for cat_label, cat_id in catalogs.items():
        res = v.query_region(c, radius=15*u.arcsec, catalog=cat_id)
        if not len(res):
            print(f"  {cat_label}: no match within 15 arcsec")
            continue
        t = res
        t["sep_arcsec"] = c.separation(
            SkyCoord(t["RAJ2000"], t["DEJ2000"])).arcsec
        t.sort("sep_arcsec")
        print(f"  {cat_label}: {len(t)} match(es)")
        t.pprint(max_width=-1)