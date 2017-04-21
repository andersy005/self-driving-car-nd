import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from tracking.CalibrateCamera import CalibrateCamera
from tracking.ImageFilters import ImageFilters
from tracking.Line import Line
from tracking.ProjectionManager import ProjectionManager


class RoadManager():

    """
     class that handles image, projection and line propagation pipeline decisions
    """

    def __init__(self, camCal, keepN=10, debug=False):
        """
        Initialize lineManager.
        """
        # for both left and right lines
        # set debugging
        self.debug = debug

        # frame Number
        self.currentFrame = None

        # keep last N
        self.keepN = keepN

        # copy of camera calibration parameters
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size

        # mid point
        self.mid = int(self.y/2)

        # create projection manager
        self.projMgr = ProjectionManager(camCal, keepN=keepN, debug=debug)

        # Default left-right lane masking
        self.maskDelta = 5

        # road statistics
        # the left and right lanes curvature measurement coulb be misleading -
        # need a threshold to indicate straight road.
        self.roadStraight = False

        # radius of curvature of the line in meters
        self.radiusOfCurvature = None

        # vehicle offset from the center
        self.lineBasePos = None

        # left lines only
        # left lane identifier
        self.left = 1

        # ghosting of left lane (for use in trouble spots - i.e:bridge or in
        # harder challenges)
        self.lastNLEdges = None

        # left lane line class
        self.leftLane = Line(self.left, self.x, self.y, self.maskDelta)

        # left lane stats
        self.leftLaneLastTop = None

        # right lines only
        # right lane identifier
        self.right = 2

        # ghosting of right lane (for use in trouble spots)
        self.lastNREdges = None

        # right lane line class
        self.rightLane = Line(self.right, self.x, self.y, self.maskDelta)

        # right lane stats
        self.rightLaneLastTop = None

        # road overhead and unwarped views
        self.roadsurface = np.zeros((self.y, self.x, 3), dtype=np.uint8)
        self.roadunwarped = None

        # number of points fitted
        self.leftLanePoints = 0
        self.rightLanePoints = 0

        # cloudy mode
        self.cloudyMode = False

        # pixel offset from direction of travel
        self.lastLeftRightOffset = 0

        # boosting
        self.boosting = 0.0

        # resulting image
        self.final = None

        # for debugging only
        if self.debug:
            self.diag1 = np.zeros((self.y, self.x, 3), dtype=np.float32)

    def find_lane_locations(self, projected_masked_lines):
        """
        Function to find starting lane line positions.
        Returns left and right column positions.
        """

        height = projected_masked_lines.shape[0]
        width = projected_masked_lines.shape[1]
        lefthistogram = np.sum(projected_masked_lines[int(
            height/2):height, 0:int(width/2)], axis=0).astype(np.float32)
        righthistogram = np.sum(projected_masked_lines[int(
            height/2):height, int(width/2):width], axis=0).astype(np.float32)
        leftpos = np.argmax(lefthistogram)
        rightpos = np.argmax(righthistogram) + int(width/2)
        return leftpos, rightpos, rightpos-leftpos

    def findLanes(self, img):
        if self.currentFrame is None:
            self.currentFrame = 0
        else:
            self.currentFrame += 1

        self.currentImageFtr = ImageFilters(self.projMgr.camCal, debug=True)
        self.currentImageFtr.imageQ(img)

        # detected cloudy condition
        if self.currentImageFtr.skyText == 'Sky Condition: cloudy' and self.currentFrame == 0:
            self.cloudyMode = True
        # choose a default filter based on weather condition
        # line class can update filter based on what it wants too (different
        # for each lane line).
        if self.cloudyMode:
            self.currentImageFtr.applyFilter3()
            self.maskDelta = 20
            self.leftLane.setMaskDelta(self.maskDelta)
            self.rightLane.setMaskDelta(self.maskDelta)

        elif self.currentImageFtr.skyText == 'Sky Condition: clear' or self.currentImageFtr.skyText == 'Sky Condition: tree shaded':
            self.currentImageFtr.applyFilter2()
        elif self.currentFrame < 2:
            self.currentImageFtr.applyFilter4()
        else:
            self.currentImageFtr.applyFilter5()

        self.currentImageFtr.horizonDetect(debug=True)

        # low confidence?
        if self.leftLane.confidence < 0.5 or self.rightLane.confidence < 0.5 or self.currentFrame < 2:
            self.projMgr.findInitialRoadCorners(self.currentImageFtr)
            self.initialGradient = self.projMgr.currentGradient

            # use visibility to lower the FoV
            # adjust source for perspective projection accordingly
            if self.currentImageFtr.horizonFound:
                self.roadHorizonGap = self.projMgr.currentGradient - \
                    self.currentImageFtr.roadhorizon
                newTop = self.currentImageFtr.roadhorizon + self.roadHorizonGap
                self.projMgr.setSrcTop(
                    newTop - self.currentImageFtr.visibility, self.currentImageFtr.visibility)

            self.lastNREdges = self.currentImageFtr.currentRoadEdge
            self.lastNLEdges = self.currentImageFtr.currentRoadEdge
            masked_edges = self.currentImageFtr.getProjection()

            masked_edge = masked_edges[:, :, 1]
            leftpos, rightpos, distance = self.find_lane_locations(masked_edge)
            self.leftLane.setBasePos(leftpos)
            self.leftLane.find_lane_lines_points(masked_edge)

            self.rightLane.setBasePos(rightpos)
            self.rightLane.find_lane_lines_points(masked_edge)

            self.leftLane.fitpoly()
            leftprojection = self.leftLane.applyLineMask(
                self.currentImageFtr.getProjection(self.leftLane.side))
            self.leftLane.radius_in_meters(distance)
            self.leftLane.meters_from_center_of_vehicle(distance)

            self.rightLane.fitpoly()
            rightprojection = self.rightLane.applyLineMask(
                self.currentImageFtr.getProjection(self.rightLane.side))
            self.rightLane.radius_in_meters(distance)
            self.rightLane.meters_from_center_of_vehicle(distance)

        else:
            # apply boosting ...
            # For Challenges ONLY
            if self.cloudyMode:
                if self.currentImageFtr.skyText == 'Sky Condition: cloudy':
                    self.boosting = 0.4
                    self.lastNREdges = self.currentImageFtr.miximg(
                        self.currentImageFtr.currentRoadEdge, self.lastNREdges, 1.0, 0.4)
                else:
                    self.boosting = 1.0
                    self.lastNREdges = self.currentImageFtr.miximg(
                        self.currentImageFtr.currentRoadEdge, self.lastNREdges, 1.0, 1.0)
                self.currentImageFtr.currentRoadEdge = self.lastNREdges
            elif self.currentImageFtr.skyText == 'Sky condition: surrounded by trees':
                self.boosting = 0.0

            # project the new frame to a plane for further analysis
            self.projMgr.project(self.currentImageFtr,
                                 self.lastLeftRightOffset)

            # Find approximate left right positions and distance apart
            masked_edges = self.currentImageFtr.getProjection()
            masked_edge = masked_edges[:, :, 1]

            # Left Lane Projection setup
            leftprojection = self.leftLane.applyLineMask(
                self.currentImageFtr.getProjection(self.leftLane.side))
            leftPoints = np.nonzero(leftprojection)
            self.leftLane.allX = leftPoints[1]
            self.leftLane.allY = leftPoints[0]
            self.leftLane.fitpoly2()

            # Right Lane Projection setup
            rightprojection = self.rightLane.applyLineMask(
                self.currentImageFtr.getProjection(self.rightLane.side))
            rightPoints = np.nonzero(rightprojection)
            self.rightLane.allX = rightPoints[1]
            self.rightLane.allY = rightPoints[0]
            self.rightLane.fitpoly2()

            # take and calculate some measurements
            distance = self.rightLane.pixelBasePos - self.leftLane.pixelBasePos
            self.leftLane.radius_in_meters(distance)
            self.leftLane.meters_from_center_of_vehicle(distance)
            self.rightLane.radius_in_meters(distance)
            self.rightLane.meters_from_center_of_vehicle(distance)

            leftTop = self.leftLane.getTopPoint()
            rightTop = self.rightLane.getTopPoint()

            # attempt to move up the lane lines if we missed predictions
            if self.leftLaneLastTop is not None and self.rightLaneLastTop is not None:
                # Only do this if we are certain that our visibility is good.
                if self.currentImageFtr.visibility > -30:
                    # If either lines differs by greater than 50 pixel vertically
                    # we need to request the shorter line to go higher
                    if abs(self.leftLaneLastTop[1] - self.rightLaneLastTop[1]) > 50:
                        if self.leftLaneLastTop[1] > self.rightLaneLastTop[1]:
                            self.leftLane.requestTopY(self.rightLaneLastTop[1])
                        else:
                            self.rightLane.requestTopY(self.leftLaneLastTop[1])

                    # if the lane line has fallen to below our threshold, get
                    # it to come back up
                    if leftTop is not None and leftTop[1] > self.mid - 100:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if leftTop is not None and leftTop[1] > self.leftLaneLastTop[1]:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if rightTop is not None and rightTop[1] > self.mid-100:
                        self.rightLane.requestTopY(rightTop[1]-10)
                    if rightTop is not None and rightTop[1] > self.rightLaneLastTop[1]:
                        self.rightLane.requestTopY(rightTop[1]-10)

                # visibility poor....
                # Need to be less aggressive going back up the lane...
                # let at least 30 frame pass before trying to move forward.
                elif self.currentFrame > 30:
                    # if either lines differs by greater than 50 pixel vertically
                    # we need to request the shorter line to go higher.
                    if abs(self.leftLaneLastTop[1] - self.rightLaneLastTop[1]) > 50:
                        if self.leftLaneLastTop[1] > self.rightLaneLastTop[1] and leftTop is not None:
                            self.leftLane.requestTopY(leftTop[1] - 10)
                        elif rightTop is not None:
                            self.rightLane.requestTopY(rightTop[1] - 10)

                    # if the lane has fallen to below the threshold, get it to
                    # come back up
                    if leftTop is not None and leftTop[1] > self.mid + 100:
                        self.leftLane.requestTopY(leftTop[1] - 10)
                    if leftTop is not None and leftTop[1] > self.leftLaneLastTop[1]:
                        self.leftLane.requestTopY(leftTop[1] - 10)

                    if rightTop is not None and rightTop[1] > self.mid + 100:
                        self.rightLane.requestTopY(rightTop[1] - 10)
                    if rightTop is not None and rightTop[1] > self.rightLaneLastTop[1]:
                        self.rightLane.requestTopY(rightTop[1] - 10)

        # Update Stats and Top points for next frame.
        self.leftLaneLastTop = self.leftLane.getTopPoint()
        self.rightLaneLastTop = self.rightLane.getTopPoint()
        self.leftLanePoints = len(self.leftLane.allX)
        self.rightLanePoints = len(self.rightLane.allX)

        # update road statistics for display
        self.lineBasePos = (self.leftLane.lineBasePos +
                            self.rightLane.lineBasePos)
        if self.leftLane.radiusOfCurvature > 0.0 and self.rightLane.radiusOfCurvature > 0.0:
            self.radiusOfCurvature = (
                self.leftLane.radiusOfCurvature + self.rightLane.radiusOfCurvature)/2.0
            if self.leftLane.radiusOfCurvature > 3000.0:
                self.roadStraight = True
            elif self.rightLane.radiusOfCurvature > 3000.0:
                self.roadStraight = True
            else:
                self.roadStraight = False
        elif self.leftLane.radiusOfCurvature < 0.0 and self.rightLane.radiusOfCurvature < 0.0:
            self.radiusOfCurvature = (
                self.leftLane.radiusOfCurvature + self.rightLane.radiusOfCurvature)/2.0
            if self.leftLane.radiusOfCurvature < -3000.0:
                self.roadStraight = True
            elif self.rightLane.radiusOfCurvature < -3000.0:
                self.roadStraight = True
            else:
                self.roadStraight = False
        else:
            self.roadStraight = True

        # create road mask polygon for reprojection back onto perspective view
        roadpoly = np.concatenate(
            (self.rightLane.XYPolyline, self.leftLane.XYPolyline[::-1]), axis=0)
        roadmask = np.zeros((self.y, self.x), dtype=np.uint8)
        cv2.fillConvexPoly(roadmask, roadpoly, 64)
        self.roadsurface[:, :, 0] = self.currentImageFtr.miximg(
            leftprojection, self.leftLane.linemask, 0.5, 0.3)
        self.roadsurface[:, :, 1] = roadmask
        self.roadsurface[:, :, 2] = self.currentImageFtr.miximg(
            rightprojection, self.rightLane.linemask, 0.5, 0.3)

        # unwarp the roadsurface
        self.roadunwarped = self.projMgr.curUnWarp(
            self.currentImageFtr, self.roadsurface)

        # Create the final image
        self.final = self.currentImageFtr.miximg(
            self.currentImageFtr.currentImage, self.roadunwarped, 0.95, 0.75)

        # draw dots ad polyline
        if self.debug:
            font = cv2.FONT_HERSHEY_COMPLEX

            self.diag1, M = self.projMgr.unwarp_lane(self.currentImageFtr.makefull(
                self.projMgr.diag1), self.projMgr.currentSrcRoadCorners, self.projMgr.currentDstRoadCorners, self.projMgr.mtx)
            self.diag1 = np.copy(self.projMgr.diag4)
            self.leftLane.scatter_plot(self.diag1)
            self.leftLane.polyline(self.diag1)
            self.rightLane.scatter_plot(self.diag1)
            self.rightLane.polyline(self.diag1)
            cv2.putText(self.diag1, 'Frame: %d' %
                        (self.currentFrame), (30, 30), font, 1, (255, 0, 0), 2)

            self.leftLane.scatter_plot(self.projMgr.diag4)
            self.leftLane.polyline(self.projMgr.diag4)
            self.rightLane.scatter_plot(self.projMgr.diag4)
            self.rightLane.polyline(self.projMgr.diag4)

            cv2.putText(self.projMgr.diag4, 'Frame: %d' %
                        (self.currentFrame), (30, 30), font, 1, (255, 255, 0), 2)
            cv2.putText(self.projMgr.diag4, 'Left: %d count,  %4.1f%% confidence, detected: %r' % (
                self.leftLanePoints, self.leftLane.confidence*100, self.leftLane.detected), (30, 60), font, 1, (255, 255, 0), 2)
            cv2.putText(self.projMgr.diag4, 'Left: RoC: %fm, DfVC: %fcm' % (
                self.leftLane.radiusOfCurvature, self.leftLane.lineBasePos*100), (30, 90), font, 1, (255, 255, 0), 2)

            cv2.putText(self.projMgr.diag4, 'Right %d count,  %4.1f%% confidence, detected: %r' % (
                self.rightLanePoints, self.rightLane.confidence*100, self.rightLane.detected), (30, 120), font, 1, (255, 255, 0), 2)
            cv2.putText(self.projMgr.diag4, 'Right RoC: %fm, DfVC: %fcm' % (
                self.rightLane.radiusOfCurvature, self.rightLane.lineBasePos*100), (30, 150), font, 1, (255, 255, 0), 2)

            if self.boosting > 0.0:
                cv2.putText(self.projMgr.diag4, 'Boosting @ %f%%' % (
                    self.boosting), (30, 180), font, 1, (128, 128, 192), 2)

            self.projMgr.diag4 = self.currentImageFtr.miximg(
                self.projMgr.diag4, self.roadsurface, 1.0, 2.0)
            self.projMgr.diag2 = self.currentImageFtr.miximg(
                self.projMgr.diag2, self.roadunwarped, 1.0, 0.5)
            self.projMgr.diag1 = self.currentImageFtr.miximg(
                self.projMgr.diag1, self.roadunwarped[self.mid:self.y, :, :], 1.0, 2.0)

    def drawLaneStats(self, color=(224, 192, 0)):
        font = cv2.FONT_HERSHEY_COMPLEX
        if self.roadStraight:
            cv2.putText(self.final, 'Estimated lane curvature: road nearly straight',
                        (30, 60), font, 1, color, 2)
        elif self.radiusOfCurvature > 0.0:
            cv2.putText(self.final, 'Estimated lane curvature: center is %fm to the right' % (
                self.radiusOfCurvature), (30, 60), font, 1, color, 2)
        else:
            cv2.putText(self.final, 'Estimated lane curvature: center is %fm to the left' %
                        (-self.radiusOfCurvature), (30, 60), font, 1, color, 2)

        if self.lineBasePos < 0.0:
            cv2.putText(self.final, 'Estimated left of center: %5.2fcm' %
                        (-self.lineBasePos*100), (30, 90), font, 1, color, 2)
        elif self.lineBasePos > 0.0:
            cv2.putText(self.final, 'Estimated right of center: %5.2fcm' % (
                self.lineBasePos*100), (30, 90), font, 1, color, 2)
        else:
            cv2.putText(self.final, 'Estimated at center of road',
                        (30, 90), font, 1, color, 2)
