import requests, io
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

adql = """
SELECT main_id, ra, dec, rvz_redshift, rvz_radvel, otype, morphtype, galdim_majaxis, galdim_minaxis
FROM basic
WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 54.125, -53.3333, 0.75)) = 1
  AND rvz_redshift BETWEEN 0.043 AND 0.088
ORDER BY rvz_redshift ASC
"""

host = "simbad.cds." + "unistra.fr"
url = "https://" + host + "/simbad/sim-tap/sync"
r = requests.post(url, data={"request": "doQuery", "lang": "adql", "format": "votable", "query": adql}, timeout=30)

tbl = Table.read(io.BytesIO(r.content), format="votable")
df = tbl.to_pandas()
coords = SkyCoord(df["ra"].values * u.deg, df["dec"].values * u.deg)
df["RA_hms"] = coords.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
df["DEC_dms"] = coords.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)
f_rest = 1.420405752
df["HI_Freq_GHz"] = (f_rest / (1.0 + df["rvz_redshift"])).round(5)
df["Orig_Chan"] = ((3.124 - df["HI_Freq_GHz"]) / 0.001 + 1.0).round(1)

cols = ["main_id", "RA_hms", "DEC_dms", "rvz_redshift", "rvz_radvel", "HI_Freq_GHz", "Orig_Chan", "morphtype", "galdim_majaxis", "galdim_minaxis"]
print(f"Found {len(df)} spectroscopic filament galaxies in Zone 1:\n")
print(df[cols].to_string(index=False))
df.to_csv("zone1_optical_filament_galaxies.csv", index=False)
print("\nSaved to zone1_optical_filament_galaxies.csv")
