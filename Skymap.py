from skyfield.api import load
from skyfield.data import hipparcos
from astropy.coordinates import *
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def get_stars(time: Time, location: EarthLocation) -> np.array:
    with load.open(hipparcos.URL) as f:
        stars_raw = hipparcos.load_dataframe(f)

    data = [[],[],[]]

    for index, star in stars_raw.iterrows():
        if star["magnitude"] > 3:
            continue

        altaz = SkyCoord(
            star["ra_degrees"],
            star["dec_degrees"],
            unit=u.deg
        ).transform_to(AltAz(location=location, obstime=time))

        alt = float((altaz.alt*u.deg).value)
        az = float((altaz.az*u.deg).value)

        if alt < 0:
            continue

        data[0].append(star["magnitude"])
        data[1].append(alt)
        data[2].append(az)

    return np.array(data)

utcoffset = -4 * u.hour #EST
#time = Time.now() - utcoffset
t = Time(f"2025-2-7 00:00:00") - utcoffset
observer = EarthLocation(lat=38.9864 * u.deg, lon=-76.8657 * u.deg, height=60)
stars = get_stars(t, observer)

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
circle = plt.Circle((0, 0), 1, transform=ax.transData._b, color="black")
ax.add_artist(circle)

print(stars[1])

ax.grid(False)
ax.set_ylim(0,1)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.scatter(
    stars[2],
    np.tan(np.pi / 4 - np.deg2rad(stars[1]) / 2),
    s=100 * 10 ** (stars[0] / -2.512),
    c="white",
    marker  =".",
    linewidths=0,
    zorder=2
)
plt.show()