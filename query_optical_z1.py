from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
import pandas as pd
import numpy as np

center = SkyCoord("03h36m30s -53d20m00s", frame="icrs")
f_rest = 1.420405752  # GHz
rows = []

print("1. Querying VizieR (6dF Galaxy Survey VII/259)...")
v = Vizier(columns=["6dFGS", "RAJ2000", "DEJ2000", "cz", "q_cz"], row_limit=-1)
v.TIMEOUT = 60
viz_res = v.query_region(center, radius=45 * u.arcmin, catalog="VII/259/spectra")
if len(viz_res) > 0:
    df_6df = viz_res[0].to_pandas()
    for _, r in df_6df.iterrows():
        cz = pd.to_numeric(r.get("cz"), errors="coerce")
        if pd.notna(cz) and 13000 <= cz <= 26500:
            z = cz / 299792.458
            freq = f_rest / (1.0 + z)
            chan = (3.124 - freq) / 0.001 + 1.0
            c = SkyCoord(f"{r['RAJ2000']} {r['DEJ2000']}", unit=(u.hourangle, u.deg))
            rows.append({
                "Catalog": "6dFGS",
                "Name": str(r.get("6dFGS", "")),
                "RA_hms": c.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True),
                "DEC_dms": c.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True),
                "RA_deg": round(c.ra.deg, 5),
                "DEC_deg": round(c.dec.deg, 5),
                "Redshift": round(z, 6),
                "Vel_kms": round(cz, 1),
                "HI_Freq_GHz": round(freq, 5),
                "Orig_Chan": round(chan, 1),
                "Inc_deg": np.nan,
            })

print("2. Querying VizieR (HyperLEDA VII/237 for inclinations & extra redshifts)...")
v2 = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "v", "logd25", "logr25"], row_limit=-1)
v2.TIMEOUT = 60
leda_res = v2.query_region(center, radius=45 * u.arcmin, catalog="VII/237/pgc")
if len(leda_res) > 0:
    df_leda = leda_res[0].to_pandas()
    for _, r in df_leda.iterrows():
        cz = pd.to_numeric(r.get("v"), errors="coerce")
        if pd.notna(cz) and 13000 <= cz <= 26500:
            z = cz / 299792.458
            freq = f_rest / (1.0 + z)
            chan = (3.124 - freq) / 0.001 + 1.0
            c = SkyCoord(f"{r['RAJ2000']} {r['DEJ2000']}", unit=(u.hourangle, u.deg))
            logr = pd.to_numeric(r.get("logr25"), errors="coerce")
            # Standard Hubble inclination formula from axis ratio b/a = 10^(-logr25)
            inc = np.nan
            if pd.notna(logr) and logr >= 0:
                ba = 10.0 ** (-logr)
                q0 = 0.2
                cos2_i = max(0.0, (ba**2 - q0**2) / (1.0 - q0**2))
                inc = round(np.degrees(np.arccos(np.sqrt(cos2_i))), 1)
            rows.append({
                "Catalog": "HyperLEDA",
                "Name": str(r.get("Name", "")),
                "RA_hms": c.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True),
                "DEC_dms": c.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True),
                "RA_deg": round(c.ra.deg, 5),
                "DEC_deg": round(c.dec.deg, 5),
                "Redshift": round(z, 6),
                "Vel_kms": round(cz, 1),
                "HI_Freq_GHz": round(freq, 5),
                "Orig_Chan": round(chan, 1),
                "Inc_deg": inc,
            })

out_df = pd.DataFrame(rows)
if len(out_df) > 0:
    out_df = out_df.sort_values("HI_Freq_GHz", ascending=False)
    print(f"\nFound {len(out_df)} spectroscopic filament galaxies inside your HI cube:\n")
    print(out_df.to_string(index=False))
    out_df.to_csv("zone1_optical_filament_galaxies.csv", index=False)
    print("\nSaved to zone1_optical_filament_galaxies.csv")
else:
    print("No matches returned—check internet access or widen redshift range.")
