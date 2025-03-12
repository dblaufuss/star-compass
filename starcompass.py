from skyfield.api import load
from skyfield.data import hipparcos
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, get_body
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import astropy.units as u

from astropy.time import Time
from datetime import datetime
import time as time_lib

from scipy.optimize import brute
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture

import exifread
import rawpy
import cv2
import os
import json

def get_bright_stars(time: Time, location: EarthLocation, magnitude: float = 4, cam_fov: float = 122.8) -> pd.DataFrame:
    with load.open(hipparcos.URL) as f:
        stars_raw = hipparcos.load_dataframe(f).sort_values(by=["magnitude"])

    data = {
        "magnitude": [],
        "radius": [],
        "theta": []
    }

    for index, star in stars_raw.iterrows():
        if star["magnitude"] > magnitude or np.isnan(star["magnitude"]):
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

        if alt < (180-cam_fov)/2:
            continue

        data["magnitude"].append(star["magnitude"])
        data["radius"].append(np.cos(np.deg2rad(alt)))
        data["theta"].append(np.deg2rad(az)+np.pi/2)

    return pd.DataFrame(data)

def get_celestial_bodies(time: Time, location: EarthLocation, magnitude: float = 4, cam_fov: float = 122.8) -> pd.DataFrame:
    bodies = ["moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
    
    #https://promenade.imcce.fr/en/pages5/572.html
    magnitudes = [-12.7, -2.2, -4.6, -2.3, -2.7, -0.4, 5.7, 7.9]

    data = {
        "magnitude": [],
        "radius": [],
        "theta": [],
    }

    for body, magnitude in zip(bodies, magnitudes):
        if magnitude > magnitude:
            continue

        altaz = get_body(body, time).transform_to(AltAz(location=location, obstime=time))

        alt = float((altaz.alt*u.deg).value)
        az = float((altaz.az*u.deg).value)

        if alt < (180-cam_fov)/2:
            continue
        
        data["magnitude"].append(magnitude)
        data["radius"].append(np.cos(np.deg2rad(alt)))
        data["theta"].append(np.deg2rad(az)+np.pi/2)

    return pd.DataFrame(data)
    
def get_visible_objects(time: Time, location: EarthLocation, magnitude: float = 4, cam_fov: float = 122.8) -> pd.DataFrame:
    return pd.concat([
        get_bright_stars(time, location, magnitude, cam_fov),
        get_celestial_bodies(time, location, magnitude, cam_fov)
    ]).sort_values(by="magnitude")

def read_image(img_src: str) -> tuple[np.array, Time]:
    img = cv2.cvtColor(rawpy.imread(img_src).raw_image_visible, cv2.COLOR_BAYER_RGGB2GRAY)
    
    with open(img_src, "rb") as f:
        exif = exifread.process_file(f)

    time_str = str(exif["Image DateTime"])
    utcoffset = time_lib.timezone * u.second
    time = Time(datetime.strptime(time_str, r"%Y:%m:%d %H:%M:%S"), format="datetime") + utcoffset

    return img, time

def find_stars(img: np.array, max_num: int = None, cam_fov: float = 122.8) -> pd.DataFrame:
    mean, median, std = sigma_clipped_stats(img)
    
    daofind = DAOStarFinder(3*std, 15, brightest=max_num)
    stars = daofind(img-median).to_pandas().sort_values(by="daofind_mag")

    stars["ycentroid"] = stars["ycentroid"].apply(lambda y: y - img.shape[0]/2)
    stars["xcentroid"] = stars["xcentroid"].apply(lambda x: x - img.shape[1]/2)

    radius = np.sqrt(stars["xcentroid"]**2 + stars["ycentroid"]**2)
    radius = radius/np.max(radius) * np.cos(np.deg2rad((180-cam_fov)/2))

    #https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#declination
    with open("declinationData.json", "r") as f:
        mag_dec = float(json.load(f)["result"][0]["declination"])

    theta = -np.arctan2(stars["ycentroid"], stars["xcentroid"]) + np.deg2rad(mag_dec)

    return pd.DataFrame({
        "magnitude": stars["daofind_mag"],
        "radius": radius,
        "theta": theta
    })

def find_nearest(object: pd.Series, bodies: pd.DataFrame) -> float:
    close = None
    
    for index, row in bodies.iterrows():
        distance = np.sqrt(object["radius"]**2 + row["radius"]**2 - 2*object["radius"]*row["radius"]*np.cos(object["theta"] - row["theta"]))
        if close is None or distance < close:
            close = distance

    return close

def least_squares(rotation: float, found: pd.DataFrame, bodies: pd.DataFrame, error: float = 2.0) -> float:
    sum_ls = 0

    for index, row in found.iterrows():
        new_row = row.copy()
        new_row["theta"] = new_row["theta"] + np.deg2rad(rotation)
        min_distance = find_nearest(new_row, bodies)
        sum_ls += (min_distance**2)/error**2

    return sum_ls

def find_heading(img_src: str, location: EarthLocation) -> float:
    img, time = read_image(img_src)

    bodies = get_visible_objects(time, location)
    found = find_stars(img)

    ls = lambda x: least_squares(x, found, bodies)
    heading = brute(ls, [(0, 359)])

    return heading[0]

def plot_skymap(time: Time, location: EarthLocation, cam_fov: float = 122.8, found_overlay = False, img: np.array = None, found_rotation: float = 0) -> plt.Figure:
    plt.close("all")
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    max_radius = np.cos(np.deg2rad((180-cam_fov)/2))
    ax.set_ylim(0, max_radius)
    circle = plt.Circle((0, 0), max_radius, transform=ax.transData._b, color="gray")
    ax.add_artist(circle)

    ax.annotate(
        text="N",
        xy=(-np.pi*3/2,max_radius-0.01),
        xytext=(-np.pi*3/2,max_radius+0.07),
        horizontalalignment="center",
        verticalalignment="bottom",
        arrowprops=dict(facecolor="black", arrowstyle="<|-")
    )
    ax.annotate(
        text="E",
        xy=(0,max_radius-0.01),
        xytext=(0,max_radius+0.07),
        horizontalalignment="left",
        verticalalignment="center",
        arrowprops=dict(facecolor="black", arrowstyle="<|-")
    )
    ax.annotate(
        text="S",
        xy=(-np.pi/2,max_radius-0.01),
        xytext=(-np.pi/2,max_radius+0.07),
        horizontalalignment="center",
        verticalalignment="top",
        arrowprops=dict(facecolor="black", arrowstyle="<|-")
    )
    ax.annotate(
        text="W",
        xy=(-np.pi,max_radius-0.01),
        xytext=(-np.pi,max_radius+0.07),
        horizontalalignment="right",
        verticalalignment="center",
        arrowprops=dict(facecolor="black", arrowstyle="<|-")
    )

    objects = get_visible_objects(time, location)

    ax.scatter(
        objects["theta"],
        objects["radius"],
        s=20 * 3 ** (objects["magnitude"] / -2.512),
        c="white",
        marker=".",
        linewidths=0,
        zorder=2
    )

    if found_overlay:
        found = find_stars(img)

        ax.scatter(
            found["theta"]+np.deg2rad(found_rotation),
            found["radius"],
            s=20,
            marker="o",
            facecolor="none",
            edgecolors="red"
        )

        ax.text(
            -np.pi/2,
            max_radius+0.25,
            f"φ = {found_rotation:.2f}°",
            horizontalalignment="center"
        )

        #fig.suptitle("Skymap with Rotated Found Stars")

    #else:
        #fig.suptitle("Skymap Made with Star Catalog")

    fig.tight_layout()

    return fig

def plot_found_stars(img: np.array) -> plt.Figure:
    plt.close("all")
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.axis("off")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    norm = simple_norm(img, "linear", percent=99.9)
    ax.imshow(img, cmap="Greys_r", norm=norm)
    #ax.set_title("Raw Camera Image with Found Stars")

    mean, median, std = sigma_clipped_stats(img)
    daofind = DAOStarFinder(3*std, 15)
    stars = daofind(img-median).to_pandas()

    positions = np.transpose((stars["xcentroid"], stars["ycentroid"]))
    apertures = CircularAperture(positions, r=4.0)

    apertures.plot(color="red", lw=5, alpha=0.5)
    
    fig.tight_layout()

    return fig

def plot_least_squares(img: np.array, time: Time, location: EarthLocation) -> plt.Figure:
    plt.close("all")
    
    rotations = np.arange(0, 361, 1, dtype=float)
    results = []

    found = find_stars(img)
    bodies = get_visible_objects(time, location)

    for rot in rotations:
        results.append(least_squares(rot, found, bodies))
    
    fig, ax = plt.subplots()
    ax.plot(rotations, results)

    ax.set_xlabel("φ (°)")
    ax.set_ylabel("Alignment Least Square")

    #ax.set_title("Least Squares Alignment for Rotations")

    fig.tight_layout()

    return fig

if __name__ == "__main__":
    observer = EarthLocation(lat=39.006543*u.deg, lon=-76.866053*u.deg, height=46*u.meter)
    
    '''
    path = "images/east/DSC_5221.NEF"
    img, t = read_image(path)
    res = find_heading(path, observer)
    print(res)
    
    skymap = plot_skymap(t, observer)
    skymap.savefig("out/skymap.png", dpi=500)

    skymap_overlay = plot_skymap(t, observer, found_overlay=True, img=img, found_rotation=res)
    skymap_overlay.savefig("out/skymap_overlay.png", dpi=500)

    found_plot = plot_found_stars(img)
    found_plot.savefig("out/found_stars.png", dpi=500)

    ls_plot = plot_least_squares(img, t, observer)
    ls_plot.savefig("out/least_squares.png", dpi=500)
    
    '''
    directions = ["north", "east", "south", "west"]
    data = dict.fromkeys(directions)

    for direction in directions:
        print(direction)
        data[direction] = []
        for file in os.listdir(f"images/{direction}"):
            heading = find_heading(f"images/{direction}/{file}", observer)
            data[direction].append(heading)
            print(file, heading)

    data = pd.DataFrame(data)
    print(data)
    data.to_csv("out/res.csv", index=False)