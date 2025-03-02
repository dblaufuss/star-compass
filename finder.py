from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from photutils.detection import DAOStarFinder
from matplotlib import pyplot as plt
import numpy as np
import cv2
import rawpy

raw = rawpy.imread("/home/deb/Pictures/NIKOND3100/DSC_5049.NEF").raw_image_visible
img = cv2.cvtColor(raw, cv2.COLOR_BAYER_RGGB2GRAY)

#img = cv2.cvtColor(cv2.imread("calibresult.png"), cv2.COLOR_RGB2GRAY)


mean, median, std = sigma_clipped_stats(img)

print(mean, median, std)

daofind = DAOStarFinder(5*std, 10)

sources = daofind(img-median)

print(sources)

norm = simple_norm(img, "sqrt", percent=99.9)

plt.imshow(img, cmap="Greys_r", norm=norm)
plt.colorbar()
plt.scatter(sources["xcentroid"], sources["ycentroid"], s=50, marker="o", facecolor="none", edgecolors="red")
plt.show()