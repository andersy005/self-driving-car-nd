import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from advanced.CalibrateCamera import CalibrateCamera
from advanced.ImageFilters import ImageFilters


class ProjectionManager():
    """ Class that handles the projection of lane lines detected."""

    def __init__(self, camCal, keepN=10, gradientLevel=75, debug=False):
        # set debugging
        self.debug = debug

        # frameNumber
        self.currentFrame = None

        # keep last N
        self.KeepN = keepN

        # Keep our own copy of the camera calibration
        self.camCal = camCal

        # copy of camera calibration parameters
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size

        # Projection mask calculations
        self.xbottom1 = int(self.x/16)
        self.xbottom2 = int(self.x*15 / 16)
        self.xtop1 = int(self.x*14/32)
        self.xtop2 = int(self.x*18/32)
        self.ybottom1 = self.y
        self.ybottom2 = self.y
        self.ytopbox = int(self.y*9/16)

        # mid point in picture (by height)
        self.mid = int(self.y/2)

        # ghosting
        self.roadGhost = np.zeros((self.mid, self.x), dtype=np.uint8)

        # gradient level starts here
        self.gradient0 = self.mid + gradientLevel

        # current image filter
        self.currentImageFilter = None

        # current road corners
        self.currentSrcRoadCorners = None

        # current horizon
        self.currentHorizon = True

        # current gradient
        self.currentGradient = None

        # last n projected image filters
        self.recentProjected = []

        # last n road corners
        self.recentRoadCorners = []

        # last n horizon detected
        self.recentHorizon = []

        # last n gradient detected
        self.recentGradient = []

        # generate destination rect for projection of road to flat plane
        us_lane_width = 12   # US highway width: 12 feet wide
        approx_dest = 20     # Approximate distance to vanishing point from end of rectangle
        scale_factor = 15    # scaling for display
        top = approx_dest * scale_factor
        left = -us_lane_width/2*scale_factor
        right = us_lane_width/2*scale_factor
        self.currentDstRoadCorners = np.float32([[(self.x/2) + left, top], [(
            self.x/2)+right, top], [(self.x/2)+right, self.y], [(self.x/2)+left, self.y]])

        # set up debugging diag screens
        if self.debug:
            self.diag1 = np.zeros((self.mid, self.x, 3), dtype=np.float32)
            self.diag2 = np.zeros((self.y, self.x, 3), dtype=np.float32)
            self.diag3 = np.zeros((self.y, self.x, 3), dtype=np.float32)
            self.diag4 = np.zeros((self.y, self.x, 3), dtype=np.float32)

    def region_of_interest(self, img, vertices):
        """ Function that creates a region of interest mask.
        Only keeps the region of the image defined by the polygon formed from
        'vertices'. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # Defining a 3 channel or 1 channel color to fill the mask depending on
        # the input image
        if len(img.shape) > 2:
            channel_count = img.shap[2]  # i.e 3 or 4 depending on the image
            ignore_mask_color = (255,)*channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill
        # color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # Returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def draw_area_of_interest(self, img, areas, color=[128, 0, 128], thickness=2):
        """ Draw outline of given area"""
        for points in areas:
            for i in range(len(points)-1):
                cv2.line(img, (points[i][0], points[i][1]), (points[
                         i+1][0], points[i+1][1]), color, thickness)
            cv2.line(img, (points[0][0], points[0][1]), (points[
                     len(points)-1][0], points[len(points)-1][1]), color, thickness)

    def draw_area_of_interest_for_projection(self, img, areas, color=[128, 0, 128], thickness1=2, thickness2=10):
        for points in areas:
            for i in range(len(points)-1):
                if i == 0 or i == 1:
                    cv2.line(img, (points[i][0], points[i][1]), (points[
                             i+1][0], points[i+1][1]), color, thickness1)
                else:
                    cv2.line(img, (points[i][0], points[i][1]), (points[
                             i+1][0], points[i+1][1]), color, thickness2)
            cv2.line(img, (points[0][0], points[0][1]), (points[
                     len(points)-1][0], points[len(points)-1][1]), color, thickness1)

    def draw_masked_area(self, img, areas, color=[128, 0, 128], thickness=2):
        for points in areas:
            for i in range(len(points)-1):
                cv2.line(img, (points[i][0], points[i][1]), (points[
                         i+1][0], points[i+1][1]), color, thickness)
            cv2.line(img, (points[0][0], points[0][1]), (points[
                     len(points)-1][0], points[len(points)-1][1]), color, thickness)

    def draw_bounding_box(self, img, boundingbox, color=[0, 255, 0], thickness=6):
        x1, y1, x2, y2 = boundingbox
        cv2.line(img, (x1, y1), (x2, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y2), color, thickness)
        cv2.line(img, (x2, y2), (x1, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y1), color, thickness)

    def draw_parallel_lines_pre_projection(self, img, lane_info, color=[128, 0, 0], thickness=5):
        """draw parallel lines in a perspective image that will later be projected into a flat surface
        """
        lx1 = lane_info[3][0]
        rx1 = lane_info[4][0]
        rx2 = lane_info[5][0]
        lx2 = lane_info[6][0]
        ly1 = lane_info[3][1]
        ry1 = lane_info[4][1]
        ry2 = lane_info[5][1]
        ly2 = lane_info[6][1]
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=6, backoff=0, debug=False):
        """
        calculate and draw initial estimated lines on the roadway.
        """
        if backoff == 0:
            backoff = thickness*5
        ysize = img.shape[0]
        midleft = img.shape[1]/2-200+backoff*2
        midright = img.shape[1]/2+200-backoff*2
        top = ysize/2+backoff*2
        rightslopemin = 0.5  # 8/backoff
        rightslopemax = 3.0  # backoff/30
        leftslopemax = -0.5  # -8/backoff
        leftslopemin = -3.0  # -backoff/30
        try:
            # rightline and leftline cumlators
            rl = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
            ll = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = ((y2-y1)/(x2-x1))
                    sides = (x1+x2)/2
                    vmid = (y1+y2)/2
                    if slope > rightslopemin and slope < rightslopemax and sides > midright and vmid > top:   # right
                        if debug:
                            #print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                            cv2.line(img, (x1, y1), (x2, y2),
                                     [128, 128, 0], thickness)
                        rl['num'] += 1
                        rl['slope'] += slope
                        rl['x1'] += x1
                        rl['y1'] += y1
                        rl['x2'] += x2
                        rl['y2'] += y2
                    elif slope > leftslopemin and slope < leftslopemax and sides < midleft and vmid > top:   # left
                        if debug:
                            #print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                            cv2.line(img, (x1, y1), (x2, y2),
                                     [128, 128, 0], thickness)
                        ll['num'] += 1
                        ll['slope'] += slope
                        ll['x1'] += x1
                        ll['y1'] += y1
                        ll['x2'] += x2
                        ll['y2'] += y2

            if rl['num'] > 0 and ll['num'] > 0:
                # average/extrapolate all of the lines that makes the right
                # line
                rslope = rl['slope']/rl['num']
                rx1 = int(rl['x1']/rl['num'])
                ry1 = int(rl['y1']/rl['num'])
                rx2 = int(rl['x2']/rl['num'])
                ry2 = int(rl['y2']/rl['num'])

                # average/extrapolate all of the lines that makes the left line
                lslope = ll['slope']/ll['num']
                lx1 = int(ll['x1']/ll['num'])
                ly1 = int(ll['y1']/ll['num'])
                lx2 = int(ll['x2']/ll['num'])
                ly2 = int(ll['y2']/ll['num'])

                # find the right and left line's intercept, which means solve the following two equations
                # rslope = ( yi - ry1 )/( xi - rx1)
                # lslope = ( yi = ly1 )/( xi - lx1)
                # solve for (xi, yi): the intercept of the left and right lines
                # which is:  xi = (ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope)
                # and        yi = ry2 + rslope*(xi-rx2)
                xi = int((ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope))
                yi = int(ry2 + rslope*(xi-rx2))

                # calculate backoff from intercept for right line
                if rslope > rightslopemin and rslope < rightslopemax:   # right
                    ry1 = yi + int(backoff)
                    rx1 = int(rx2-(ry2-ry1)/rslope)
                    ry2 = ysize-1
                    rx2 = int(rx1+(ry2-ry1)/rslope)
                    cv2.line(img, (rx1, ry1), (rx2, ry2),
                             [255, 0, 0], thickness)

                # calculate backoff from intercept for left line
                if lslope < leftslopemax and lslope > leftslopemin:   # left
                    ly1 = yi + int(backoff)
                    lx1 = int(lx2-(ly2-ly1)/lslope)
                    ly2 = ysize-1
                    lx2 = int(lx1+(ly2-ly1)/lslope)
                    cv2.line(img, (lx1, ly1), (lx2, ly2),
                             [255, 0, 0], thickness)

                # if we have all of the points - draw the backoff line near the
                # horizon
                if lx1 > 0 and ly1 > 0 and rx1 > 0 and ry1 > 0:
                    cv2.line(img, (lx1, ly1), (rx1, ry1),
                             [255, 0, 0], thickness)

            # return the left and right line slope, found rectangler box shape
            # and the estimated vanishing point.
            return lslope+rslope, lslope, rslope, (lx1, ly1), (rx1, ry1), (rx2, ry2), (lx2, ly2), (xi, yi)
        except:
            return -1000, 0.0, 0.0, (0, 0), (0, 0), (0, 0), (0, 0)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, backoff=0, debug=False):
        """
        generate a set of hough lines and calculates its estimates for lane lines.
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn using the new single line for left and right lane line method.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
            []), minLineLength=min_line_len, maxLineGap=max_line_gap)
        masked_lines = np.zeros(img.shape, dtype=np.uint8)
        lane_info = self.draw_lines(
            masked_lines, lines, backoff=backoff, debug=debug)

        return masked_lines, lane_info

    # function to project the undistorted camera image to a plane looking down.
    def unwarp_lane(self, img, src, dst, mtx):
        # Pass in your image, 4 source points src = np.float32([[,],[,],[,],[,]])
        # and 4 destination points dst = np.float32([[,],[,],[,],[,]])
        # Note: you could pick any four of the detected corners
        # as long as those four corners define a rectangle
        # One especially smart way to do this would be to use four well-chosen
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        # use cv2.warpPerspective() to warp your image to a top-down view
        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        # warped = gray
        return warped, M

    # function to find starting lane line positions
    # return left and right column positions
    def find_lane_locations(self, masked_lines):
        height = masked_lines.shape[0]
        width = masked_lines.shape[1]
        lefthistogram = np.sum(
            masked_lines[int(height/2):height, 0:int(width/2)], axis=0).astype(np.float32)
        righthistogram = np.sum(masked_lines[int(
            height/2):height, int(width/2):width], axis=0).astype(np.float32)
        leftpos = np.argmax(lefthistogram)
        rightpos = np.argmax(righthistogram)+int(width/2)
        # print("leftpos",leftpos,"rightpos",rightpos)
        return leftpos, rightpos, rightpos-leftpos

    # hough version1
    def hough_lines1(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi/180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        min_line_length = 120  # 50 75 25 minimum number of pixels making up a line
        max_line_gap = 40    # 40 50 20 maximum gap in pixels between connectable line segments
        return self.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=30, debug=debug)

    # hough version2
    def hough_lines2(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi/180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        min_line_length = 75  # 50 75 25 minimum number of pixels making up a line
        max_line_gap = 40    # 40 50 20 maximum gap in pixels between connectable line segments
        return self.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=40, debug=debug)

    # hough version3
    def hough_lines3(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi/180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        min_line_length = 25  # 50 75 25 minimum number of pixels making up a line
        max_line_gap = 20    # 40 50 20 maximum gap in pixels between connectable line segments
        return self.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=50, debug=debug)

    # hough version4
    def hough_lines4(self, masked_edges, debug=False):
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi/180  # angular resolution in radians of the Hough grid
        # minimum number of votes (intersections in Hough grid cell)
        threshold = 40
        min_line_length = 20  # 50 75 25 minimum number of pixels making up a line
        max_line_gap = 20    # 40 50 20 maximum gap in pixels between connectable line segments
        return self.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=50, debug=debug)

    def findInitialRoadCorners(self, imgftr):
        """ Function to find initial road corners to find a projection matrix.
        Once corners are found, project the edges into a plane or when we fall below 50% confidence
        in the lane line detected.
        """
        # first time?
        if self.currentFrame == None:
            self.currentFrame = 0
        else:
            self.currentFrame += 1

        # piece together images that we want to project
        edge = imgftr.edges()[:, :, 0]

        # we are defining a four sided polygon to mask
        vertices = np.array([[(self.xbottom1, self.ybottom1), (self.xtop1, self.ytopbox), (
            self.xtop2, self.ytopbox), (self.xbottom2, self.ybottom2)]], dtype=np.int32)

        # Now mask it
        masked_edge = self.region_of_interest(np.copy(edge), vertices)
        masked_edges = np.dstack((edge, masked_edge, masked_edge))

        # Cascading hough mapping line attempts
        hough = 1
        line_image, lane_info = self.hough_lines1(masked_edge)
        if lane_info[0] == -1000:
            hough = 2
            line_image, lane_info = self.hough_lines2(masked_edge)
            if lane_info[0] == -1000:
                hough = 3
                line_image, lane_info = self.hough_lines3(masked_edge)
                if lane_info[0] == -1000:
                    hough = 4
                    line_image, lane_info = self.hough_lines4(masked_edge)

        # if we made it: calculate the area of interest
        if lane_info[0] > -1000:
            self.currentGradient = lane_info[3][1]

            areaOfInterest = np.array([[(lane_info[3][0]-50, lane_info[3][1]-11),
                                        (lane_info[4][0]+50,
                                         lane_info[4][1]-11),
                                        (lane_info[4][0]+525,
                                         lane_info[4][1]+75),
                                        (lane_info[4][0]+500, lane_info[5][1]),
                                        (lane_info[4][0]-500, lane_info[6][1]),
                                        (lane_info[3][0]-525, lane_info[3][1]+75)]], dtype=np.int32)
            # generate src rect for projection of road to flat plane
            self.currentSrcRoadCorners = np.float32(
                [lane_info[3], lane_info[4], lane_info[5], lane_info[6]])

            # generate grayscaled map image
            projected_roadsurface, M = self.unwarp_lane(np.copy(
                masked_edges), self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)
            imgftr.setProjection(projected_roadsurface)
            self.lane_info = lane_info

        # create debug/diag screens if required
        if self.debug:
            # diag 1 screen - road edges with masked out area shown
            self.diag1 = imgftr.makehalf(masked_edges)*4

            # rest is only valid if we are able to get lane_info...
            if lane_info[0] > -1000:
                leftbound = int(lane_info[7][0]-(self.x*0.1))
                rightbound = int(lane_info[7][0]+(self.x*0.1))
                topbound = int(lane_info[7][1]-(self.y*0.15))
                bottombound = int(lane_info[7][1]+(self.y*0.05))
                boundingbox = (leftbound-2, topbound-2,
                               rightbound+2, bottombound+2)

                # non-projected image with found points
                ignore = np.copy(line_image)*0
                self.diag2 = imgftr.miximg(imgftr.currentImage, masked_edges*2)
                self.diag2 = imgftr.miximg(
                    self.diag2, np.dstack((line_image, ignore, ignore)))
                if imgftr.visibility > -30:
                    self.draw_masked_area(self.diag2, vertices)
                self.draw_bounding_box(self.diag2, boundingbox)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(self.diag2, 'Frame: %d   Hough: %d' % (
                    self.currentFrame, hough), (30, 30), font, 1, (255, 0, 0), 2)
                self.draw_area_of_interest(self.diag2, areaOfInterest, color=[
                                           0, 128, 0], thickness=5)

                cv2.putText(self.diag2, 'x1,y1: %d,%d' % (int(lane_info[3][0]), int(lane_info[3][
                            1])), (int(lane_info[3][0])-250, int(lane_info[3][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag2, 'x2,y2: %d,%d' % (int(lane_info[4][0]), int(lane_info[4][
                            1])), (int(lane_info[4][0]), int(lane_info[4][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag2, 'x3,y3: %d,%d' % (int(lane_info[5][0]), int(lane_info[5][
                            1])), (int(lane_info[5][0])-200, int(lane_info[5][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag2, 'x4,y4: %d,%d' % (int(lane_info[6][0]), int(lane_info[6][
                            1])), (int(lane_info[6][0])-200, int(lane_info[6][1])-30), font, 1, (255, 0, 0), 2)

                # diag 3 screen - complete road RGB image projected
                diag3tmp = imgftr.miximg(imgftr.currentImage, masked_edges*4)
                self.draw_area_of_interest_for_projection(diag3tmp, areaOfInterest, color=[
                                                          0, 128, 0], thickness1=1, thickness2=50)
                self.draw_parallel_lines_pre_projection(
                    diag3tmp, lane_info, color=[128, 0, 0], thickness=2)
                self.diag3, M = self.unwarp_lane(
                    diag3tmp, self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)

                cv2.putText(self.diag3, 'x1,y1: %d,%d' % (int(self.currentDstRoadCorners[0][0]), int(self.currentDstRoadCorners[0][
                            1])-1), (int(self.currentDstRoadCorners[0][0])-300, int(self.currentDstRoadCorners[0][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag3, 'x2,y2: %d,%d' % (int(self.currentDstRoadCorners[1][0]), int(self.currentDstRoadCorners[1][
                            1])-1), (int(self.currentDstRoadCorners[1][0]), int(self.currentDstRoadCorners[1][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag3, 'x3,y3: %d,%d' % (int(self.currentDstRoadCorners[2][0]), int(self.currentDstRoadCorners[2][
                            1])-1), (int(self.currentDstRoadCorners[2][0]), int(self.currentDstRoadCorners[2][1])-30), font, 1, (255, 0, 0), 2)
                cv2.putText(self.diag3, 'x4,y4: %d,%d' % (int(self.currentDstRoadCorners[3][0]), int(self.currentDstRoadCorners[3][
                            1])-1), (int(self.currentDstRoadCorners[3][0])-300, int(self.currentDstRoadCorners[3][1])-30), font, 1, (255, 0, 0), 2)

                # diag 4 screen - road edges with masked out area shown
                # projected
                self.diag4, M = self.unwarp_lane(imgftr.makefull(
                    self.diag1), self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)

    def project(self, imgftr, leftRightOffset=0):
        """ Function that projectes the edges into a plane.
        This function is for when we are now at confidence greater than 50% in the lane line detected.
        """
        self.currentFrame += 1
        lane_info = self.lane_info

        # piece together images that we want to project
        edge = imgftr.edges()[:, :, 0]

        # we are defining a four sided polygon to mask
        vertices = np.array([[(self.xbottom1, self.ybottom1), (self.xtop1, self.ytopbox), (
            self.xtop2, self.ytopbox), (self.xbottom2, self.ybottom2)]], dtype=np.int32)

        # Now mask it
        masked_edge = self.region_of_interest(np.copy(edge), vertices)
        masked_edges = np.dstack((edge, masked_edge, masked_edge))
        hough = 0

        # calculate the area of interest
        self.currentGradient = lane_info[3][1]
        areaOfInterest = np.array([[(lane_info[3][0]-50, lane_info[3][1]-11),
                                    (lane_info[4][0]+50,
                                     lane_info[4][1]-11),
                                    (lane_info[4][0]+525,
                                     lane_info[4][1]+75),
                                    (lane_info[4][0]+500, lane_info[5][1]),
                                    (lane_info[4][0]-500, lane_info[6][1]),
                                    (lane_info[3][0]-525, lane_info[3][1]+75)]], dtype=np.int32)

        # generate src rect for projection of road to flat plane since this is a fast version of the projector
        # in general we will use the last projection information what we obtained from the last search.
        # however, roadmanager can reset this based on gap thresholds with the last detected horizon.
        # see setSrcTop() function below.
        self.currentSrcRoadCorners = np.float32(
            [lane_info[3], lane_info[4], lane_info[5], lane_info[6]])

        # generate grayscaled map image
        projected_roadsurface, M = self.unwarp_lane(np.copy(
            masked_edges), self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)
        imgftr.setProjection(projected_roadsurface)

        # Create debug/diag screens if required
        if self.debug:
            # diag 1 screen - road edges with masked out area shown
            self.diag1 = imgftr.makehalf(masked_edges) * 4

            # rest is only valid if we are able to get lane_info....
            if lane_info[0] > -1000:
                leftbound = int(lane_info[3][0]-50+leftRightOffset)
                rightbound = int(lane_info[4][0]+50+leftRightOffset)
                topbound = int(lane_info[3][1]-71)
                bottombound = int(lane_info[3][1]-11)
                boundingbox = (leftbound-2, topbound-2,
                               rightbound+2, bottombound+2)

                # non projected image with found points
                self.diag2 = imgftr.miximg(imgftr.currentImage, masked_edges*2)
                if imgftr.visibility > -30:
                    self.draw_masked_area(self.diag2, vertices)
                self.draw_bounding_box(self.diag2, boundingbox)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(self.diag2, 'Frame: %d   Hough: %d' % (
                    self.currentFrame, hough), (30, 30), font, 1, (255, 0, 0), 2)
                self.draw_area_of_interest(self.diag2, areaOfInterest, color=[
                                           0, 128, 0], thickness=5)

                # diag 3 screen - complete road RGB image projected
                diag3tmp = imgftr.miximg(imgftr.currentImage, masked_edges*4)
                self.draw_area_of_interest_for_projection(diag3tmp, areaOfInterest, color=[
                                                          0, 128, 0], thickness1=1, thickness2=50)
                self.draw_parallel_lines_pre_projection(
                    diag3tmp, lane_info, color=[128, 0, 0], thickness=2)
                self.diag3, M = self.unwarp_lane(
                    diag3tmp, self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)

                # diag 4 screen - road edges with masked out area shown
                # projected
                self.diag4, M = self.unwarp_lane(imgftr.makefull(
                    self.diag1), self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)

    def curWarp(self, imgftr, image):
        """
        warp the perspective view to planar view
        """
        warped, M = self.unwarp_lane(
            image, self.currentSrcRoadCorners, self.currentDstRoadCorners, self.mtx)
        return warped

    def curUnWarp(self, imgftr, image):
        """
        unwarp the planar view back to perspective view
        """
        unwarped, M = self.unwarp_lane(
            image, self.currentDstRoadCorners, self.currentSrcRoadCorners, self.mtx)
        return unwarped

    def setSrcTop(self, newTop, sideDelta):
        """
        an attempt to dampen the bounce of the car and the road surface.
        called by RoadManager class
        """

        if newTop > self.gradient0:
            self.ytopbox = newTop-15
            self.xtop1 += sideDelta
            self.xtop2 -= sideDelta
            self.lane_info = (self.lane_info[0],
                              self.lane_info[1],
                              self.lane_info[2],
                              (self.lane_info[3][0]+sideDelta, newTop),
                              (self.lane_info[4][0]-sideDelta, newTop),
                              self.lane_info[5],
                              self.lane_info[6],
                              self.lane_info[7])

    def setSrcTopX(self, sideDelta):
        """another attempt to dampen the bounce of the car and the road surface.
        """
        self.lane_info = (self.lane_info[0],
                          self.lane_info[1],
                          self.lane_info[2],
                          (self.xtop1+sideDelta, self.lane_info[3][1]),
                          (self.xtop2+sideDelta,
                           self.lane_info[4][1]),  self.lane_info[5],
                          self.lane_info[6], (self.lane_info[7][0]+sideDelta, self.lane_info[7][1]))
