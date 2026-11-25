import requests, io
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np

url = "https://tapvizier.cds." + "unistra.fr/TAPVizieR/tap/sync"

adql = """
SELECT "6dFGS", RAJ2000, DEJ2000, cz, q_cz
FROM "VII/259/spectra"
WHERE 1=CONTAINS(POINT('ICRS', RAJ2000, DEJ2000), CIRCLE('ICRS', 54.125, -53.3333, 0.75))
  AND cz BETWEEN 13000 AND 26500
ORDER BY cz ASC
"""

r = requests.post(url, data={"request": "doQuery", "lang": "adql", "format": "votable", "query": adql}, timeout=30)
print("HTTP status:", r.status_code)
tbl = Table.read(io.BytesIO(r.content), format="votable")
df = tbl.to_pandas()

coords = SkyCoord(df["RAJ2000"].values * u.deg, df["DEJ2000"].values * u.deg)
df["RA_hms"] = coords.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
df["DEC_dms"] = coords.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)
df["Redshift"] = (df["cz"] / 299792.458).round(6)
f_rest = 1.420405752
df["HI_Freq_GHz"] = (f_rest / (1.0 + df["Redshift"])).round(5)
df["Orig_Chan"] = ((3.124 - df["HI_Freq_GHz"]) / 0.001 + 1.0).round(1)

print(f"\nFound {len(df)} 6dFGS filament galaxies in Zone 1:\n")
print(df[["6dFGS", "RA_hms", "DEC_dms", "cz", "Redshift", "HI_Freq_GHz", "Orig_Chan"]].to_string(index=False))
df.to_csv("zone1_optical_filament_galaxies.csv", index=False)
print("\nSaved to zone1_optical_filament_galaxies.csv")
