import time
from astroquery.ipac.ned import Ned

hosts = ["WISEA J033706.65-534512.5",   # ATCA_18 host
         "WISEA J033555.94-534510.8"]   # ATCA_20 host

want = ["Published Velocity", "Published Velocity Uncertainty",
        "Published Redshift", "Published Redshift Uncertainty",
        "Refcode", "Qualifiers", "Spectrograph",
        "Measurement Mode Technique"]

for name in hosts:
    print("=" * 70)
    print(name)
    z = None
    for attempt in range(3):
        try:
            z = Ned.get_table(name, table="redshifts")
            break
        except Exception as e:
            print(f"attempt {attempt + 1} failed:", repr(e))
            time.sleep(5)
    if z is None:
        print("giving up on", name)
        continue
    cols = [c for c in want if c in z.colnames]
    z[cols].pprint_all()