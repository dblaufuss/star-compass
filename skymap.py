from skyfield.api import load
from skyfield.data import hipparcos
from astropy.coordinates import *
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import time

MAX_MAGNITUDE = 4.5
CAM_FOV = 104.1
MIN_ALTITUDE = (180-CAM_FOV)/2

def get_stars(time: Time, location: EarthLocation) -> np.array:
    with load.open(hipparcos.URL) as f:
        stars_raw = hipparcos.load_dataframe(f)

    data = [[],[],[]]

    for index, star in stars_raw.iterrows():
        if star["magnitude"] > MAX_MAGNITUDE or np.isnan(star["magnitude"]):
            continue

        if np.isnan(star["ra_degrees"]) or np.isnan(star["dec_degrees"]):
            continue

        altaz = SkyCoord(
            star["ra_degrees"],
            star["dec_degrees"],
            unit=u.deg
        ).transform_to(AltAz(location=location, obstime=time))

        alt = float((altaz.alt*u.deg).value)
        az = float((altaz.az*u.deg).value)

        if alt < MIN_ALTITUDE:
            continue

        if star["magnitude"] < 0:
            data[0].append(0)
        else:
            data[0].append(star["magnitude"])
        
        data[1].append(alt)
        data[2].append(az)

    return np.array(data)


def get_bodies(time: Time, location: EarthLocation) -> np.array:
    bodies = ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    #https://promenade.imcce.fr/en/pages5/572.html
    magnitudes = [-12.7, -2.2, -4.6, -2.3, -2.7, -0.4, 5.7, 7.9]

    data = [[],[],[]]

    for body, magnitude in zip(bodies, magnitudes):
        if magnitude > MAX_MAGNITUDE:
            continue

        altaz = get_body(body, t).transform_to(AltAz(location=location, obstime=time))

        alt = float((altaz.alt*u.deg).value)
        az = float((altaz.az*u.deg).value)

        if alt < MIN_ALTITUDE:
            continue
        
        data[0].append(magnitude)
        data[1].append(alt)
        data[2].append(az)

    return np.array(data)

utcoffset = time.timezone * u.second
t = Time(f"2025-2-21 22:35:40") + utcoffset
#t = Time.now()

observer = EarthLocation(lat=38.911262702627, lon=-78.88359672603178, height=100)
#observer = EarthLocation(lat=39.006543*u.deg, lon=-76.866053*u.deg, height=60)

stars = get_stars(t, observer)
bodies = get_bodies(t, observer)

MAX_RADIUS = np.tan(np.pi/4 - np.deg2rad(MIN_ALTITUDE)/2)

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
circle = plt.Circle((0, 0), MAX_RADIUS, transform=ax.transData._b, color="black")
ax.add_artist(circle)

ax.grid(False)
ax.set_ylim(0, MAX_RADIUS)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax.scatter(
    np.deg2rad(stars[2])+np.pi/2,
    np.tan(np.pi / 4 - np.deg2rad(stars[1]) / 2),
    s=5 * 1 ** (stars[0] / -2.512),
    c="white",
    marker=".",
    linewidths=0,
    alpha=1-0.8*(stars[0]/4.5),
    zorder=2
)

ax.scatter(
    np.deg2rad(bodies[2])+np.pi/2,
    np.tan(np.pi / 4 - np.deg2rad(bodies[1]) / 2),
    s=5 * 2 ** (bodies[0] / -2.512),
    c="red",
    marker=".",
    linewidths=0,
    zorder=2
)

#ax.set_title(f"{t}\nLAT {round(observer.lat.value, 4)}°, LON {round(observer.lon.value, 4)}°\nSIMULATED FOV: {CAM_FOV}°")
plt.tight_layout()
plt.savefig("chart.png", dpi=1000)
plt.show()