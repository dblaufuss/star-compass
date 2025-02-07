from skyfield.api import load
from skyfield.data import hipparcos
from astropy.coordinates import *
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

time = Time.now()
sun = get_sun(time)

observer = EarthLocation(lat=38.9864*u.deg, lon=-76.8657*u.deg)

sun_coord = SkyCoord(sun.ra, sun.dec)

test_coord = SkyCoord(320.69621276, -15.35417026, unit=u.deg)


with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

stars_mag = []
stars_alt = []
stars_az = []

for index, star in stars.iterrows():
    if star["magnitude"] >= 3.5:
        continue

    altaz = SkyCoord(star["ra_degrees"], star["dec_degrees"], unit=u.deg).transform_to(AltAz(location=observer, obstime=time))
    alt = float((altaz.alt*u.deg).value)
    az = float((altaz.az*u.deg).value)

    if alt < 0:
        continue

    stars_mag.append(star["magnitude"])
    stars_alt.append(alt)
    stars_az.append(az)

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

circle = plt.Circle((0, 0), 1, transform=ax.transData._b, color="black")
ax.add_artist(circle)

ax.scatter(
    stars_az,
    np.tan(np.pi/4 - np.deg2rad(np.array(stars_alt))/2),
    s=100 * 10 ** (np.array(stars_mag) / -2.5),
    c="white",
    marker=".",
    linewidths=0,
    zorder=2
)

plt.axis('off')
plt.show()