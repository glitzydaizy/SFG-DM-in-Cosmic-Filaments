from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned

v = Vizier(columns=["*"], row_limit=10,
           vizier_server="vizier.cfa.harvard.edu")

sources = {
    "source1": "03h37m04.692s -53d01m21.42s",
    "source2": "03h37m05.876s -52d58m20.33s"
}

vizier_cats = {
    "DES_DR2": "II/371/des_dr2",
    "WISExSCOS_PhotoZ": "J/ApJS/225/5/wisc"
}

for name, pos in sources.items():
    c = SkyCoord(pos)
    print(f"\n================ {name} ({pos}) ================")
    
    # 1. Query NED (all literature redshifts & object types)
    try:
        ned_res = Ned.query_region(c, radius=10*u.arcsec)
        print(f"  NED: {len(ned_res)} match(es)")
        if len(ned_res):
            ned_res["Object Name", "RA", "DEC", "Type", "Velocity", "Redshift", "Separation"].pprint(max_width=-1)
    except Exception as e:
        print(f"  NED: no match or query note ({e})")

    # 2. Query deep VizieR optical & photo-z catalogs
    for label, cat_id in vizier_cats.items():
        res = v.query_region(c, radius=10*u.arcsec, catalog=cat_id)
        if not len(res):
            print(f"  {label}: no match within 10 arcsec")
            continue
        t = res
        print(f"  {label}: {len(t)} match(es)")
        t.pprint(max_width=-1)