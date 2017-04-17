
# import some useful modules
import argparse
import sys
import re
import os
import cv2
from moviepy.editor import VideoFileClip

# 1. CalibrateCamera: class that handles camera calibration operations
from advanced.CalibrateCamera import CalibrateCamera

# 2. ImageFilters: class that handles image analysis and filtering operations
from advanced.ImageFilters import ImageFilters

# 3. ProjectionManager: class that handles projection calculations and
# operations
from advanced.ProjectionManager import ProjectionManager

# 4. Line: class that handles line detection, measurements and confidence
# calculations and operations
from advanced.Line import Line

# 5. RoadManager: class that handles image, projection and line
# propagation pipeline decisions
from advanced.RoadManager import RoadManager

# 6. DiagManager: class that handles diagnostic output requests
from advanced.DiagManager import DiagManager

# process_road_image handles rendering a single image through the pipeline


def process_road_image(img, roadMgr, diagMgr, scrType=0, debug=False):
    # Run the functions
    roadMgr.findLanes(img)

    # debug/diagnostics requested
    if debug:
        # offset for text rendering overlay
        offset = 0
        color = (192, 192, 0)
        # default - full diagnostics
        if scrType & 3 == 3:
            diagScreen = diagMgr.fullDiag()
            offset = 30
        elif scrType & 3 == 2:
            diagScreen = diagMgr.projectionDiag()
            offset = 30
        elif scrType & 3 == 1:
            diagScreen = diagMgr.filterDiag()
            offset = 30
            color = (192, 192, 192)
        if scrType & 4 == 4:
            diagScreen = diagMgr.textOverlay(
                diagScreen, offset=offset, color=color)
        result = diagScreen
    else:
        if scrType & 4 == 4:
            roadMgr.drawLaneStats()
        result = roadMgr.final
    return result


def process_image(image):
    global roadMgr
    global diagMgr
    global debug
    global scrType
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = process_road_image(
        image, roadMgr, diagMgr, scrType=scrType, debug=debug)
    return result

if __name__ == "__main__":

    # set default - final/no diagnostics
    parser = argparse.ArgumentParser(prog='advanced_lane_pipeline.py', usage='python %(prog)s [options] infilename outfilename',
                                     description='Anderson\'s Udacity SDC Project 4: Advanced Lane Finding Pipeline')
    parser.add_argument('--diag', type=int, default=0,
                        help='display diagnostics: [0=off], 1=filter, 2=proj 3=full')
    parser.add_argument('--notext', action='store_true',
                        default=False, help='do not render text overlay')
    parser.add_argument('infilename', type=str, default='project_video.mp4',
                        help='input image or video file to process')
    parser.add_argument('outfilename', type=str,
                        default='project_video_out.mp4', help='output image or video file')
    args = parser.parse_args()
    debug = False

    videopattern = re.compile("^.+\.mp4$")
    imagepattern = re.compile("^.+\.(jpg|jpeg|JPG|png|PNG)$")
    image = None
    videoin = None
    valid = False

    # set up pipeline processing options
    # if video - set up in/out videos
    if videopattern.match(args.infilename):
        if videopattern.match(args.outfilename):
            if not os.path.exists(args.infilename):
                print("Video input file: %s does not exist.  Please check and try again." % (
                    args.infilename))
                sys.exit(1)
            elif os.path.exists(args.outfilename):
                print("Video output file: %s exists.  Please remove and try again." % (
                    args.outfilename))
                sys.exit(2)
            else:
                videoin = args.infilename
                videoout = args.outfilename
                valid = True
        else:
            print("Invalid video filename extension for output.  Must end with '.mp4'")
            sys.exit(3)

    # if image - set up image processing options
    elif imagepattern.match(args.infilename):
        if imagepattern.match(args.outfilename):
            if not os.path.exists(args.infilename):
                print("Image input file: %s does not exist.  Please check and try again." % (
                    args.infilename))
                sys.exit(4)
            elif os.path.exists(args.outfilename):
                print("Image output file: %s exists.  Please remove and try again." % (
                    args.outfilename))
                sys.exit(5)
            else:
                image = cv2.cvtColor(cv2.imread(
                    args.infilename), cv2.COLOR_BGR2RGB)
                valid = True
        else:
            print(
                "Invalid image filename extension for output.  Must end with one of [jpg,jpeg,JPG,png,PNG]")
            sys.exit(6)

    # set up diagnostic pipeline options if requested
    if valid:
        scrType = args.diag
        if (scrType & 3) > 0:
            debug = True
        if not args.notext:
            scrType = scrType | 4

        # initialization
        # load or perform camera calibrations
        camCal = CalibrateCamera('camera_cal', 'camera_cal/calibrationdata.p')

        # initialize road manager and its managed pipeline components/modules
        roadMgr = RoadManager(camCal, debug=debug)

        # initialize diag manager and its managed diagnostics components
        if debug:
            diagMgr = DiagManager(roadMgr)
        else:
            diagMgr = None

        # Image only?
        if image is not None:
            print("image processing %s..." % (args.infilename))
            imageout = process_image(image)
            cv2.imwrite(args.outfilename, cv2.cvtColor(
                imageout, cv2.COLOR_RGB2BGR))
            print("done image processing %s..." % (args.infilename))

        # Full video pipeline
        elif videoin is not None and videoout is not None:
            print("video processing %s..." % (videoin))
            clip1 = VideoFileClip(videoin)
            video_clip = clip1.fl_image(process_image)
            video_clip.write_videofile(videoout, audio=False)
            print("done video processing %s..." % (videoin))
    else:
        print("error detected.  exiting.")
