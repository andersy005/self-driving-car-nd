import numpy as np
import cv2
import math


class Line():
    """ Class that receives the characteristics of each line detection"""

    def __init__(self, side, x, y, maskDelta, n=10):
        # iterations to keep
        self.n = n

        # assigned side
        self.side = side

        # dimensions
        self.x = x
        self.y = y
        self.mid = int(y/2)

        # frameNumber
        self.currentFrame = None

        # was the line detected ini the last iteration?
        self.detected = False

        # was the line detected in the last iteration?
        self.confidence = 0.0
        self.confidence_based = 0

        # polynomial coefficients averaged over the last n iterations
        self.bestFit = None

        # polynomial coefficients for the most recent fit
        self.currentFit = [np.array(False)]

        # x values of the current fitted line
        self.currentX = None

        # radius of curvature of the line in meters
        self.radiusOfCurvature = None

        # distance in meters of vehicle center from the line
        self.lineBasePos = None

        # Pixel base position
        self.pixelBasePos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allX = None

        # y values for detected line pixels
        self.allY = None

        # xy values for drawing
        self.XYPolyline = None

        # mask delta for masking points in lane
        self.maskDelta = maskDelta

        # poly for fitting new values
        self.linePoly = None

        # mask for lanes
        self.linemask = np.zeros((self.y, self.x), dtype=np.uint8)

        # road manager request
        self.newYTop = None

        self.maxsum = 0.0

    def find_lane_nearest_neighbors(self, histogram, lastpos, nneighbors):
        """ Function that finds lane line positions given histogram row,
        last column positions and n_neighbors and returns column positions
        """

        ncol = len(histogram) - 1
        x = []
        alist = {"count": 0, "position": lastpos}
        for i in range(nneighbors):
            if(lastpos + i) < len(histogram) and histogram[lastpos + i] > 0:
                x.append(lastpos+i)
                if alist['count'] < histogram[lastpos+i]:
                    alist['count'] = histogram[lastpos+i]
                    alist['position'] = lastpos+i

            if (lastpos-i) > 0 and histogram[lastpos-i] > 0:
                x.append(lastpos-i)
                if alist['count'] < histogram[lastpos-i]:
                    alist['count'] = histogram[lastpos-i]
                    alist['position'] = lastpos-i
        return alist['position'], x

    def setBasePos(self, basePos):
        """Function that sets base position"""
        self.pixelBasePos = basePos

    def find_lane_lines_points(self, masked_lines):
        """Function that finds lane lines points using a sliding window histogram given starting position
        and returns arrays x and y positions.
        """
        xval = []
        yval = []
        nrows = masked_lines.shape[0] - 1
        neighbors = 12
        pos1 = self.pixelBasePos
        start_row = nrows - 16
        for i in range(int((nrows/neighbors))):
            histogram = np.sum(
                masked_lines[start_row + 10: start_row+26, :], axis=0).astype(np.uint8)
            pos2, x = self.find_lane_nearest_neighbors(
                histogram, pos1, int(neighbors*1.3))
            y = start_row + neighbors
            for i in range(len(x)):
                xval.append(x[i])
                yval.append(y)

            start_row -= neighbors
            pos1 = pos2
        self.allX = np.array(xval)
        self.allY = np.array(yval)

    def scatter_plot(self, img, size=3):
        """ scatter plot the points"""
        if self.side == 1:
            color = (192, 128, 128)
        else:
            color=(128, 128, 192)
        xy_array = np.column_stack((self.allX, self.allY)).astype(np.int32)
        for xy in xy_array:
            cv2.circle(img, (xy[0], xy[1]), size, color, -1)

    def polyline(self, img, size=5):
        """draw fitted polyline"""
        if self.side == 1:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.polylines(img, [self.XYPolyline], 0, color, size)

    def fitpoly(self, degree=2):
        """ Fit a (default = second order) polynomial to lane line
        Used for initialization when first starting up or when lane line was lost and starting
        over."""
        if len(self.allY) > 150:
            # we need to increase our pixel count by 2 to get 100% confidence.
            # and maintain the current pixel count to keep the line detection
            self.confidence_based = len(self.allY) * 2
            self.confidence = len(self.allY) / (self.confidence_based)
            self.detected = True

            self.currentFit = np.polyfit(self.allY, self.allX, degree)
            polynomial = np.poly1d(self.currentFit)

            self.currentX = polynomial(self.allY)

            # Create linepoly
            xy1 = np.column_stack((self.currentX+30, self.allY)).astype(np.int32)
            xy2 = np.column_stack((self.currentX-30, self.allY)).astype(np.int32)
            self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

            # Create mask
            self.linemask = np.zeros_like(self.linemask)
            cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

            # Add the point at the bottom
            # Note: we walked up from the bottom - so the base point should be the
            # first point.
            allY = np.insert(self.allY, 0, self.y-1)
            allX = polynomial(allY)
            self.XYPolyline = np.column_stack((allX, allY)).astype(np.int32)

            # Create the accumulator
            self.bestFit = self.currentFit

    def fitpoly2(self, degree=2):
        """ Fit a (default=second order) polynomial to lane line.
        This version assumes that a manual fit was already done is using the previously generated
        poly to fit the current line in the new frame.
        """
        if len(self.allY) > 50:
            self.currentFit = np.polyfit(self.allY, self.allX, degree)

            # sanity check
            self.diffs = self.currentFit - self.bestFit
            if abs(sum(self.diffs)) < 150.0:
                polynomial = np.poly1d(self.currentFit)

                # Add the point at the bottom
                # Note: These points are counted by NumPy - it does topdown, so our bottom point
                # is now at the end of the list.
                x = polynomial([self.y-1])
                self.allY = np.append(self.allY, self.y-1)
                self.allX = np.append(self.allX, x[0])
                self.pixelBasePos = x[0]

                # honoring the road manager request to move higher
                # NOTE: these points are counted by numpy - it does topdown, so our top point
                # is now at the front of the list.

                if self.newYTop is not None:
                    x = polynomial([self.newYTop])
                    self.allY = np.insert(self.allY, 0, self.newYTop)
                    self.allX = np.insert(self.allX, 0, x[0])
                    self.newYTop = None

                # Fit the polynomial and generate the current fit.
                self.currentX = polynomial(self.allY)
                self.XYPolyline = np.column_stack((self.currentX, self.allY)).astype(np.int32)

                # Create linepoly
                xy1 = np.column_stack((self.currentX+self.maskDelta, self.allY)).astype(np.int32)
                xy2 = np.column_stack((self.currentX-self.maskDelta, self.allY)).astype(np.int32)
                self.linePoly = np.concatenate((xy1, xy2[::-1]), axis=0)

                # Create a mask
                self.linemask = np.zeros_like(self.linemask)
                cv2.fillConvexPoly(self.linemask, self.linePoly, 64)

                # Add to the accumulators
                self.bestFit = (self.bestFit+self.currentFit) / 2

                # Figure out the confidence level
                self.confidence = len(self.allY) / self.confidence_based
                if self.confidence > 0.5:
                    self.detected =  True
                    if self.confidence > 1.0:
                        self.confidence = 1.0
                else:
                    self.detected = False 

            else: 
                # difference check failed - need to re-initialize
                self.confidence = 0.0
                self.detected = False 
        else:
            # Not enough points - need to re-initialize
            self.confidence = 0.0
            self.detected = False 



    def applyLineMask(self, img):
        """ Apply the line masking Poly"""
        img0 = img[:, :, 1]
        masked_edge = np.copy(self.linemask).astype(np.uint8)
        masked_edge[(masked_edge>0)] = 255
        return cv2.bitwise_and(img0, img0, mask=masked_edge)


    def getTopPoint(self):
        """ Get the top point of the detected line.
        Use it to see if we lost track."""
        if len(self.allY) > 0:
            y = np.min(self.allY)
            polynomial = np.poly1d(self.currentFit)
            x = polynomial([y])
            return( x[0], y)
        else:
            return None


    def requestTopY(self, newY):
        """ Road manager request to move the line detection higher. 
        Otherwise the algorithm is lazy and will lose the entire line.
        """
        self.newYTop = newY

    def setMaskDelta(self, maskDelta):
        self.maskDelta = maskDelta

    def radius_in_meters(self, distance):

        """  Define conversions in x and y from pixels space to meters given lane line separation in pixels
        NOTE: Only do calculation if it make sense - otherwise give previous answer.
        """
        if len(self.allY) > 0 and len(self.currentX) > 0 and len(self.allY) == len(self.currentX):

            #######################################################################################################
            # Note: We are using 54 instead of 30 here since our throw for the perspective transform is much longer
            #       We estimate our throw is 54 meters based on US Highway reecommended guides for Longitudinal
            #       Pavement Markings.  See: http://mutcd.fhwa.dot.gov/htm/2003r1/part3/part3a.htm
            #       Section 3A.05 Widths and Patterns of Longitudinal Pavement Markings
            #       Guidance:
            #           Broken lines should consist of 3 m (10 ft) line segments and 9 m (30 ft) gaps,
            #           or dimensions in a similar ratio of line segments to gaps as appropriate for
            #           traffic speeds and need for delineation.
            #       We are detecting about 4 and 1/3 sets of dashed line lanes on the right side:
            #           4.33x(3+9)=4.33x12=52m
            ########################################################################################################
            ym_per_pix = 52/720 # meters per pixel in y dimension
            xm_per_pix = 3.7/distance # meters per pixel in x dimension (at the base)

            # Use the midddle point in the distance of the road instead of the base where the car is at 
            y_point = self.y/2
            fit_cr = np.polyfit(self.allY*ym_per_pix, self.currentX*xm_per_pix, 2)
            self.radiusOfCurvature = ((1+(2*fit_cr[0]*y_point + fit_cr[1])** 2)**1.5)/(2*fit_cr[0])
        return self.radiusOfCurvature

    def meters_from_center_of_vehicle(self, distance):
        """ Defines conversion in x off center from pixel space to meters given lane line 
        separation in pixels.
        """
        xm_per_pix = 3.7/distance
        pixels_off_center = int(self.pixelBasePos - (self.x/2))
        self.lineBasePos = xm_per_pix * pixels_off_center
        return self.lineBasePos


           







