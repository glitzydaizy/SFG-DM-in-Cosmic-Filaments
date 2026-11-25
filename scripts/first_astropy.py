from astropy.coordinates import SkyCoord
import astropy.units as u

orion = SkyCoord("05h35m17.3s", "-05d23m28s")

print("RA in hours:", orion.ra)
print("RA in degrees:", orion.ra.deg)

print("Dec:", orion.dec)

galactic = orion.galactic

print("\nGalactic longitude:", galactic.l)
print("Galactic latitude:", galactic.b)
