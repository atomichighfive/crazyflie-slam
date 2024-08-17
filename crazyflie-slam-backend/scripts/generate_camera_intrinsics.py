import argparse
import numpy as np
import cv2 as cv
import logging
import sys
import os

from pathlib import Path
 

def main(input_dir, output_dir, width, height, preview, logger):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width*height,3), np.float32)
    objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = sorted(input_dir.glob("*.png"))
    
    image_shape = None

    for fname in images:
        logger.debug(fname)
        img = cv.imread(fname)
        if image_shape is None:
            image_shape = img.shape
        else:
            assert image_shape == img.shape
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(
            gray,
            (width,height),
            None,
            cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE
        )

        corners2 = cv.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners2)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
        else:
            logger.warning(f"Failed to find points for {fname}")
    logger.info(f"Found checkerboard in {len(objpoints)}/{len(images)} images.")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    logger.info(f"total error: {mean_error/len(objpoints)}")

    os.makedirs(output_dir, exist_ok=False)
    np.save(output_dir/"camera_matrix.npy", mtx)
    np.save(output_dir/"distortion_coefficients.npy", dist)
    np.save(output_dir/"rotation_vectors.npy", rvecs)
    np.save(output_dir/"translation_vectors.npy", tvecs)
    np.save(output_dir/"image_shape.npy", image_shape)
    logger.info(f"Wrote camera properties to {output_dir}")

    if preview:
        for fname, corners in zip(images, imgpoints):
            img = cv.imread(fname)
            h, w = img.shape[:2]
            img = cv.drawChessboardCorners(img, (width,height), corners, None)
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imshow(str(fname), img)
            cv.waitKey(0)
            cv.destroyAllWindows()



if __name__ == "__main__":
    logger=logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description='Generate Camera Intrinsics from a folder with 10 images of a checkerboard.')
    parser.add_argument("-i", "--input_dir", type=Path, help="input folder")
    parser.add_argument("-o", "--output_dir", type=Path, help="output folder")
    parser.add_argument("-x", "--width", type=int, default=6, help="Number of horizontal interior corners of the checkerboard pattern")
    parser.add_argument("-y", "--height", type=int, default=9, help="Number of vertical interior corners of the checkerboard pattern")
    parser.add_argument("--preview", action='store_true', default=False, help="Preview reprojections.")
    args = parser.parse_args()

    try:
        main(**vars(args), logger=logger)
    except Exception as e:
        logging.shutdown()
        raise e
