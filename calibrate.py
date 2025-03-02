import rawpy
import cv2 as cv
import numpy as np
import glob
from PIL import Image

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
images = glob.glob("checkerboard/*.NEF")
#images = ["checkerboard/DSC_5066.NEF"]
 
for fname in images:
    print(fname)
    img = rawpy.imread(fname).postprocess()
    #img = cv.resize(img, (0, 0), fx=0.4, fy=0.4)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,9))
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("found")
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (6,9), corners2, ret)
        #cv.imshow('img', img)
        #cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


img = cv.imread("/home/deb/Pictures/NIKOND3100/darktable_exported/DSC_5072_01.png")
#img = rawpy.imread().postprocess()
#img = cv.resize(img, (0,0), fx=0.4, fy=0.4)

h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)