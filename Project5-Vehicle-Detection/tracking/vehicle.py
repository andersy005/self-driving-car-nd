import numpy as np
import cv2
import webcolors


class Vehicle():

    def __init__(self, ID, lanes, projMgr, roadGrid, objIdx, perspectiveImage,
                 mainLaneIdx):
        """Initialization"""
        self.vehIdx = ID
        self.vehStr = '{}'.format(ID)
        self.projMgr = projMgr
        self.roadGrid = roadGrid
        self.projectedX = projMgr.projectedX
        self.projectedY = projMgr.projectedY
        self.middle = self.projectedX / 2
        self.x = projMgr.x
        self.y = projMgr.y
        self.lanes = lanes
        self.mainLaneIdx = mainLaneIdx
        self.selfieX = 640
        self.selfieY = 240

        # special effects
        # closing circle sweep

        self.sweepDone = False
        self.sweepDeltaFrame = 0

        # Scanning Sweep
        self.scanDone = False
        self.scanDeltaFrame = 0

        # this would be the width, height, depth of the vehicle
        # if we could see it in birds-eye view
        # we will calculate this during 3D reconstruction
        self.boundingShape = np.array([0.0, 0.0, 0.0]).astype(np.float32)

        # estimated x, y location in birds-ey projected view
        self.xcenter = 0
        self.ycenter = 0

        # estimated initial size 64x64
        self.deltaX = 32
        self.deltaY = 32

        # Use the projection manager's estimated height - our z value
        self.z = projMgr.z * 1.2

        # Initial windows during detection
        self.lastObjList = roadGrid.getObjectList(objIdx)
        self.initialWindows = roadGrid.getObjectListWindows(objIdx)

        # windows
        self.windows = roadGrid.getFoundAndNotOccludedWindowsInObject(objIdx)

        # boxes
        self.boxes = roadGrid.getFoundAndNotOccludedWindowsInObject(objIdx)

        # lane and location in the voxel grid teh vehicle is on
        if len(self.boxes) > 0:
            self.lane, self.yidx = roadGrid.gridCoordinates(self.boxes[0])
            self.box = self.boxes[0]
            self.xcenter, self.ycente = self.windowCenter(
                roadGrid.getBoxWindow(self.box))

        else:
            self.lane = None
            self.yidx = None
            self.box = None
            self.initialMaskVector = None

        # was the vehicle detected in the last iteration?
        self.detected = False

        # Percentage confidence
        self.detectConfidence = 0.0
        self.detectConfidence_base = 0.0
        self.initFrames = 0
        self.graceFrames = 10
        self.exitFrames = 0
        self.traveled = False

        # contour of vehicle
        self.contourInPerspective = None

        # mask of vehicle
        self.maskedProfile = None
        self.vehicleHeatMap = np.zeros(
            (self.selfieY, self.selfieX), dtype=np.float32)
        self.vehicleMaskInPespective = None

        # vehicle status and stats
        self.vehicleClassified = False
        self.color = (0, 0, 0)
        self.colorpoints = 0
        self.webColorName = None
        self.statusColor = None
        self.status = "Not Found"
        self.vehicleInLane = None
        self.previousboxes = []

        # could be one of:
        # DetectionPhase:
        #     0:Initialized
        #     1:DetectionConfirmed
        # TrackingPhase:
        #     2:Scanning
        #     3:VehicleAcquired
        #     4:VehicleLocked
        #     5:VehicleOccluded
        #     6:VehicleLeaving
        #     7:VehicleLosted

        self.mode = 0

        # array of 3d and 2d points for bounding cube
        # do the calculations for the 2d and 3d bounding box
        self.cube3d, self.cube2d = self.calculateRoughBoundingCubes(
            self.windows)

        # Create the rough masked image for projection
        self.maskVertices, self.maskedImage = self.calculateMask(
            np.copy(perspectiveImage))

        # Project the image for verification
        self.selfie = self.takeProfileSelfie(self.maskedImage)

    def updateVehicle(self, roadGrid, perspectiveImage, x=None, y=None, lane=None):
        """ Update vehicle status before tracking."""
        self.roadGrid = roadGrid
        if lane is not None:
            self.lane = lane

        # lane and location in the voxel grid the vehicle is on
        if x is not None and y is not None and self.lane is not None:
            self.ycenter = y
            self.xcenter = self.lanes[self.lane].calculateXCenter(y)
            self.window = ((self.xcenter - self.deltaX, self.ycenter - self.deltaY),
                           (self.xcenter + self.deltaX, self.ycenter + self.deltaY))

            self.windows = [self.window]
            if lane is not None:
                self.lane = lane
            if self.lane is not None:
                yidx = self.roadGrid.calculateObjectPosition(
                    self.lane, self.ycenter)

                if yidx > 0:
                    self.yidx = yidx

            self.box = self.roadGrid.getKey(self.lane, self.yidx)
            self.boxes = [self.box]
            self.roadGrid.insertTrackedObject(
                self.lane, self.yidx, self.window, self.vehIdx, tracking=True)

        elif self.mode > 2 and self.mode < 7:
            # for testing without tracking.
            self.xcenter = self.lanes[self.lane].calculateXCenter(self.ycenter)
            self.window = ((self.xcenter - self.deltaX, self.ycenter - self.deltaY),
                           (self.xcenter + self.deltaX, self.ycenter + self.deltaY))
            self.windows = [self.window]
            if lane is not None:
                self.lane = lane
            if self.lane is not None:
                yidx = self.roadGrid.calculateObjectPosition(
                    self.lane, self.ycenter)
                if yidx > 0:
                    self.yidx = yidx
            newbox = self.roadGrid.getKey(self.lane, self.yidx)

            # save last ten voxels for voxel trigger subpression
            if newbox != self.box:
                self.previousboxes.insert(0, self.box)
                self.previousboxes = self.previousboxes[:10]
                self.box = newbox
            self.boxes = [self.box]
            for oldbox in self.previousboxes:
                self.roadGrid.setOccluded(oldbox)
            self.roadGrid.insertTrackedObject(
                self.lane, self.yidx, self.window, self.vehIdx, tracking=True)

        else:
            # initial windows during detection
            if self.vehStr in self.roadGrid.vehicle_list:
                self.box = self.roadGrid.vehicle_list[self.vehStr]

            else:
                self.roadGrid.vehicle_list[self.vehStr] = self.box

            # windows
            self.windows = self.roadGrid.getFoundAndNotOccludedWindowsInVehicle(
                self.vehIdx)

            # boxes
            self.boxes = self.roadGrid.getFoundAndNotOccludedWindowsInVehicle(
                self.vehIdx)

            if len(self.boxes) > 0:
                self.lane, self.yidx = self.roadGrid.gridCoordinates(self.box)
                self.xcenter, self.ycenter = self.windowCenter(
                    self.roadGrid.getBoxWindow(self.box))

        # was the vehicle detected in the last iteration?
        self.detected = True

        # This is automatic now. Voxel will reject if not found.
        if self.mode == 0:
            self.mode = 1

        # array of 3d and 2d points for bounding cube
        # do the calculations for the 2d and 3d bounding box
        self.cube3d, self.cube2d = self.calculateRoughBoundingCubes(
            self.windows)

        # create the rough masked image for projection.
        self.maskVertices, self.maskedImage = self.calculateMask(
            np.copy(perspectiveImage))

        # project the image for verification
        self.selfie = self.takeProfileSelfie(self.maskedImage)

        return self.roadGrid

    def closest_colour(self, requested_colour):
        """ classify the vehicle by its main color components"""
        min_colours = {}
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = name

        return min_colours[min(min_colours.keys())]
