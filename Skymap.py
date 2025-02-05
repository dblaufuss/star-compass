from datetime import datetime, timezone
import pytz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from skyfield.api import Star, load, wgs84
from skyfield.data import hipparcos
from skyfield.projections import build_stereographic_projection
import Sensors

eph = load("de421.bsp")

with load.open(hipparcos.URL) as f:
    stars = hipparcos.load_dataframe(f)

location = Sensors.get_location("ws://localhost:8080")
lat = location["latitude"]
lon = location["longitude"]
print("LATITUDE:", lat)
print("LONGITUDE:", lon)

local_time = datetime.now()
utc_time = local_time.astimezone(pytz.utc)
ts = load.timescale()
t = ts.from_datetime(utc_time)
print("\nLOCAL TIME:", local_time)
print("UTC TIME:", utc_time)

sun = eph['sun']
earth = eph['earth']

observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon).at(t)
position = observer.from_altaz(alt_degrees=90, az_degrees=0)

ra, dec, distance = observer.radec()
center_object = Star(ra=ra, dec=dec)

center = earth.at(t).observe(center_object)
projection = build_stereographic_projection(center)
field_of_view_degrees = 180.0

star_positions = earth.at(t).observe(Star.from_dataframe(stars))
stars['x'], stars['y'] = projection(star_positions)

chart_size = 10
max_star_size = 100
limiting_magnitude = 10

bright_stars = (stars.magnitude <= limiting_magnitude)
magnitude = stars['magnitude'][bright_stars]

fig, ax = plt.subplots(figsize=(chart_size, chart_size))
    
border = plt.Circle((0, 0), 1, color='black', fill=True)
ax.add_patch(border)

marker_size = max_star_size * 10 ** (magnitude / -2.5)

ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars],
           s=marker_size, color='white', marker='.', linewidths=0, 
           zorder=2)

horizon = Circle((0, 0), radius=1, transform=ax.transData)
for col in ax.collections:
    col.set_clip_path(horizon)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.axis('off')

plt.show()