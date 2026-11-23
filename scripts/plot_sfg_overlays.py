import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.skyview import SkyView

hdul = fits.open("data/images_d1213/d1213_mfs.fits")
primary = next(iter(hdul))
data_2d = np.squeeze(primary.data)
wcs_2d = WCS(primary.header).celestial

bmaj_arcsec = 18.051
bmin_arcsec = 3.036
bpa_deg = -16.8

targets = [
    {
        "name": "ATCA_18_0337-5345",
        "wise": "WISEA J033706.65-534512.5",
        "coord": SkyCoord("03:37:06.66 -53:45:12.4", unit=(u.hourangle, u.deg)),
        "z": 0.075542,
        "sfr": 56
    },
    {
        "name": "ATCA_20_0335-5345",
        "wise": "WISEA J033555.94-534510.8",
        "coord": SkyCoord("03:35:55.83 -53:45:12.2", unit=(u.hourangle, u.deg)),
        "z": 0.059641,
        "sfr": 16
    }
]

for t in targets:
    c = t["coord"]
    radio_cut = Cutout2D(data_2d, position=c, size=(2.5*u.arcmin, 2.5*u.arcmin), wcs=wcs_2d)

    med = np.nanmedian(radio_cut.data)
    local_rms = 1.4826 * np.nanmedian(np.abs(radio_cut.data - med))
    levels = np.asarray((4, 6, 10, 20, 35)) * local_rms

    opt_hdu = next(iter(SkyView.get_images(position=c, survey="DSS2 Red", radius=1.25*u.arcmin)))
    ir_hdu = next(iter(SkyView.get_images(position=c, survey="DSS2 IR", radius=1.25*u.arcmin)))

    fig = plt.figure(figsize=(11, 5.2))
    panels = (
        (opt_hdu, "Optical (DSS2 Red)"),
        (ir_hdu, "Near-Infrared (DSS2 IR)")
    )

    for i, (s_hdul, label) in enumerate(panels, 1):
        s_hdu = next(iter(s_hdul))
        s_wcs = WCS(s_hdu.header)
        ax = fig.add_subplot(1, 2, i, projection=s_wcs)

        img_data = s_hdu.data
        vmin, vmax = np.percentile(img_data, (5, 99.3))
        ax.imshow(img_data, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)

        ax.contour(
            radio_cut.data,
            levels=levels,
            colors="crimson",
            linewidths=1.4,
            transform=ax.get_transform(radio_cut.wcs)
        )

        pix_scale_arcsec = abs(s_hdu.header["CDELT2"]) * 3600.0
        ny, nx = img_data.shape
        beam = Ellipse(
            xy=(0.12 * nx, 0.14 * ny),
            width=bmin_arcsec / pix_scale_arcsec,
            height=bmaj_arcsec / pix_scale_arcsec,
            angle=bpa_deg,
            edgecolor="crimson",
            facecolor="crimson",
            alpha=0.35,
            lw=1.2
        )
        ax.add_patch(beam)

        ax.set_title(f"{label}\n{t['wise']} (z = {t['z']:.4f})", fontsize=10.5)
        ax.set_xlabel("Right Ascension (J2000)", fontsize=10)
        ax.set_ylabel("Declination (J2000)" if i == 1 else "", fontsize=10)

    fig.suptitle(
        f"{t['name']} — ATCA 2.46 GHz Contours (4, 6, 10, 20, 35σ_local) | SFR ≈ {t['sfr']} M☉/yr",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_png = f"results/{t['name']}_thesis_overlay.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png} (local_rms = {local_rms*1e6:.1f} uJy/beam)")
