import requests
from io import BytesIO
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

center = SkyCoord(54.125 * u.deg, -53.3333 * u.deg)
radius = 0.75 * u.deg
f_rest = 1.420405752  # GHz

# 1. Your working HEASARC 6dFGS query
print("1. Querying HEASARC 6dFGS...")
r = requests.get(
    "https://heasarc.gsfc.nasa.gov/xamin/vo/cone",
    params={"table": "sixdfgs", "RA": 54.125, "DEC": -53.3333, "SR": 0.75},
    timeout=60,
)
t_heasarc = Table.read(BytesIO(r.content), format="votable")
df_h = t_heasarc.to_pandas()
df_h["source_cat"] = "HEASARC_6dFGS"
df_h = df_h.rename(columns={"radial_velocity": "cz", "radial_velocity_error": "e_cz"})

# 2. Partner's CfA Harvard VizieR query (6dFGS + Fleenor2005 HRS + HyperLEDA)
print("2. Querying CfA Harvard VizieR (6dFGS, Fleenor2005 HRS, HyperLEDA)...")
v = Vizier(columns=["*"], row_limit=500, vizier_server="vizier.cfa.harvard.edu")
res = v.query_region(
    center,
    radius=radius,
    catalog=["VII/259/6dfgs", "J/AJ/130/957", "VII/237/pgc"],
)

for key in res.keys():
    print(f"   -> {key}: {len(res[key])} rows")

# Filter HEASARC 6dFGS to your filament window (13,000 - 26,500 km/s)
fil = df_h[(df_h["cz"] >= 13000) & (df_h["cz"] <= 26500)].copy()
fil = fil.sort_values("cz").reset_index(drop=True)

coords = SkyCoord(fil["ra"].values * u.deg, fil["dec"].values * u.deg)
fil["RA_hms"] = coords.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True)
fil["DEC_dms"] = coords.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)
fil["z"] = (fil["cz"] / 299792.458).round(6)
fil["HI_Freq_GHz"] = (f_rest / (1.0 + fil["z"])).round(5)
fil["Cube"] = np.where(fil["HI_Freq_GHz"] >= 1.332, "part1", "part2")

cols = ["name", "RA_hms", "DEC_dms", "cz", "e_cz", "z", "HI_Freq_GHz", "Cube", "bj_mag"]
print(f"\n=== Zone 1 Filament Sample ({len(fil)} galaxies in 13,000-26,500 km/s) ===")
print(fil[cols].to_string(index=False))

fil.to_csv("zone1_optical_filament_galaxies.csv", index=False)
print("\nSaved to zone1_optical_filament_galaxies.csv")

# Also print any extra galaxies from Fleenor2005 or HyperLEDA in the same velocity slice
for cat_id in ["J/AJ/130/957", "VII/237/pgc"]:
    if cat_id in res:
        t_extra = res[cat_id].to_pandas()
        print(f"\n--- Columns in {cat_id}: {list(t_extra.columns)} ---")
        t_extra.to_csv(f"zone1_{cat_id.replace('/', '_')}.csv", index=False)
