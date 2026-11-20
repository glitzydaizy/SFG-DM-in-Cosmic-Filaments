import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
NU_OBS = 2.457   # GHz, effective MFS frequency
CAL = 0.05       # flux calibration uncertainty

# name: (z, S_obs mJy, flux error mJy)
sources = {
    "ATCA_18_0337-5345": (0.075542, 4.234, 0.224),
    "ATCA_20_0335-5345": (0.059641, 2.0, 0.6),
}

def sfr(z, S_mjy, alpha):
    dl = cosmo.luminosity_distance(z).to(u.m).value
    s14 = S_mjy * (1.4 / NU_OBS) ** (-alpha) * (1 + z) ** (alpha - 1)  # mJy
    L = 4 * np.pi * dl**2 * s14 * 1e-29                                # W/Hz
    return L, 6.35e-22 * L, 5.52e-22 * L                               # Murphy11, Bell03

# sanity check: should reproduce earlier table (55.2 and 14.6)
for name, z, S in [("ATCA_18 old", 0.075542, 4.152), ("ATCA_20 old", 0.059641, 1.789)]:
    L, m, b = sfr(z, S, 0.75)
    print(f"check {name}: L={L:.2e}  Murphy={m:.1f}  Bell={b:.1f}")
print()

for name, (z, S, eS) in sources.items():
    eS_tot = np.hypot(eS, CAL * S)
    print(f"{name}  z={z}  S={S:.3f} +/- {eS_tot:.3f} mJy")
    for alpha in (0.5, 0.75, 1.0):
        L, m, b = sfr(z, S, alpha)
        _, m_lo, _ = sfr(z, S - eS_tot, alpha)
        _, m_hi, _ = sfr(z, S + eS_tot, alpha)
        print(f"  alpha={alpha:4.2f}  L={L:.2e} W/Hz  "
              f"SFR_Murphy={m:5.1f} (+{m_hi-m:.1f}/-{m-m_lo:.1f})  SFR_Bell={b:5.1f}")
    print()