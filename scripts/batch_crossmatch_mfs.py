import re
import time
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned

# 1. Parse MIRIAD imsad output using simple whitespace splitting
sources = []
last_fitted_maj, last_fitted_min = np.nan, np.nan

with open("notes/d1213/sources_mfs_5sigma.txt") as f:
    for line in f:
        sline = line.strip()
        if sline.startswith("Fitted major/minor/pa"):
            p = sline.split()
            last_fitted_maj, last_fitted_min = float(p[2]), float(p[3])
        elif " : " in sline and re.match(r"^\d{4}[+-]\d{4}\b", sline):
            p = sline.split()
            name = p[0]
            ra, dec = p[2], p[3]
            peak_mjy = float(p[4]) * 1e3
            integ_mjy = float(p[5]) * 1e3
            flags = " ".join(p[9:])
            sources.append({
                "id": f"ATCA_{len(sources)+1:02d}_{name}",
                "ra": ra,
                "dec": dec,
                "peak_mJy": round(peak_mjy, 3),
                "int_mJy": round(integ_mjy, 3),
                "maj_arcsec": round(last_fitted_maj, 2),
                "min_arcsec": round(last_fitted_min, 2),
                "flags": flags,
                "coord": SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))
            })

print(f"Total fitted components parsed: {len(sources)}")

# 2. Filter out obvious sidelobe/edge noise-spike artifacts
clean_sources = [
    s for s in sources
    if (2.0 <= s["min_arcsec"] <= 15.0) and (5.0 <= s["maj_arcsec"] <= 40.0) and (s["int_mJy"] >= 0.35)
]
print(f"Clean compact/resolved candidates kept: {len(clean_sources)}")

# 3. Crossmatch against AllWISE (Harvard mirror) & NED
v = Vizier(columns=["AllWISE", "RAJ2000", "DEJ2000", "W1mag", "W2mag", "W3mag", "W4mag", "ccf", "qph"],
           row_limit=5, vizier_server="vizier.cfa.harvard.edu")

# WISE W4 (22 um) zero-point flux density = 8.363 Jy (Jarrett et al. 2011)
S0_W4_mJy = 8363.0

rows = []
for idx, s in enumerate(clean_sources, 1):
    c = s["coord"]
    wise_name, sep_wise = "--", np.nan
    w1, w2, w3, w4 = np.nan, np.nan, np.nan, np.nan
    w1_w2, q22, qph, ccf = np.nan, np.nan, "--", "--"
    ned_name, ned_type, z_spec = "--", "--", np.nan

    # Query AllWISE within 5 arcsec
    try:
        res = v.query_region(c, radius=5*u.arcsec, catalog="II/328/allwise")
        if len(res):
            t = res[0]
            t["sep"] = c.separation(SkyCoord(t["RAJ2000"], t["DEJ2000"])).arcsec
            t.sort("sep")
            best = t[0]
            wise_name = str(best["AllWISE"])
            sep_wise = round(float(best["sep"]), 2)
            w1, w2, w3, w4 = float(best["W1mag"]), float(best["W2mag"]), float(best["W3mag"]), float(best["W4mag"])
            w1_w2 = round(w1 - w2, 3)
            qph = str(best["qph"])
            ccf = str(best["ccf"])
            s22_mJy = S0_W4_mJy * (10 ** (-0.4 * w4))
            q22 = round(np.log10(s22_mJy / s["int_mJy"]), 2)
    except Exception as e:
        print(f"  [WISE warning {s['id']}]: {e}")

    # Query NED within 5 arcsec
    try:
        nres = Ned.query_region(c, radius=5*u.arcsec)
        if len(nres):
            nres.sort("Separation")
            nbest = nres[0]
            ned_name = str(nbest["Object Name"])
            ned_type = str(nbest["Type"])
            if not np.ma.is_masked(nbest["Redshift"]):
                z_spec = float(nbest["Redshift"])
    except Exception:
        pass

    # Preliminary classification
    if np.isnan(w1):
        status = "No_WISE_match"
    elif (not np.isnan(w1_w2) and w1_w2 > 0.8) or (not np.isnan(q22) and q22 < 0.0 and qph[-1] != "U"):
        status = "AGN_candidate"
    elif not np.isnan(q22) and q22 >= 0.0 and qph[-1] != "U":
        status = "SFG_candidate"
    else:
        status = "Uncertain_W4_upper_limit"

    rows.append((
        s["id"], s["ra"], s["dec"], s["peak_mJy"], s["int_mJy"],
        wise_name, sep_wise, round(w1, 2), round(w2, 2), round(w4, 2),
        w1_w2, q22, qph, ccf, ned_name, ned_type, z_spec, status
    ))
    print(f"[{idx}/{len(clean_sources)}] {s['id']} ({s['int_mJy']} mJy) -> {status} | q22={q22} ({qph}) | z={z_spec}")
    time.sleep(0.2)

out_table = Table(
    rows=rows,
    names=["ID", "RA", "Dec", "Peak_mJy", "Int_mJy", "AllWISE", "Sep_arcsec",
           "W1", "W2", "W4", "W1_W2", "q22", "qph", "ccf", "NED_Name", "NED_Type", "z", "Class"]
)
out_table.write("results/d1213_mfs_catalog.csv", format="csv", overwrite=True)
print("\nSaved complete catalog to results/d1213_mfs_catalog.csv!")