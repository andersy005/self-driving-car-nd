import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from tracking.CalibrateCamera import CalibrateCamera
from tracking.ImageFilters import ImageFilters
from tracking.ProjectionManager import ProjectionManager
from tracking.Line import Line


class DiagManager():
    # Initialize ImageFilter

    def __init__(self, roadManager):
        self.rMgr = roadManager
        self.pMgr = self.rMgr.projMgr

    ########################################################
    # Apply Textural Diagnostics
    ########################################################
    def textOverlay(self, diagScreen, offset, color=(64, 64, 0)):
        roadMgr = self.rMgr
        projMgr = self.pMgr
        imgFtr = self.rMgr.currentImageFtr

        font = cv2.FONT_HERSHEY_COMPLEX

        # output image stats
        y = 30 + offset
        text = '%-28s%-28s%-28sBalance: %f' % (imgFtr.skyText, imgFtr.skyImageQ,
                                               imgFtr.roadImageQ, imgFtr.roadbalance)
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        # output prjection stats in the next two lines
        y += 30
        if imgFtr.horizonFound:
            text = 'Projection Last Top: %d'
            text += '   Road Horizon: %d'
            text += '   Vanishing Point: %d'
            text = text % (roadMgr.lastTop, imgFtr.roadhorizon,
                           projMgr.lane_info[7][1])
            cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        else:
            text = 'Projection Last Top: %d   Horizon: NOT FOUND!'
            cv2.putText(diagScreen, text % (
                roadMgr.lastTop), (30, y), font, 1, color, 2)

        y += 30
        text = 'Road Backoff at: %d   Gap: %d   Visibility: %6.2fm'
        cv2.putText(diagScreen, text % (
            projMgr.currentGradient,
            projMgr.currentGradient - projMgr.gradient0,
            imgFtr.throwDistance), (30, y), font, 1, color, 2)

        # output restart stats
        y += 30
        text = 'Restart Count: %d   Projection Reset Count: %d' % (
            roadMgr.restartCount, roadMgr.resetProjectionCount)
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        # output vehicle stats
        y += 30
        text = 'Vehicle Count: %d' % (
            len(roadMgr.vehicles))
        cv2.putText(diagScreen, text, (30, y), font, 1, color, 2)

        return diagScreen

    ########################################################
    # Full diagnostics of the RoadManager
    ########################################################
    def fullDiag(self, color=(128, 128, 0)):
        roadMgr = self.rMgr
        projMgr = self.pMgr
        imgFtr = self.rMgr.currentImageFtr

        diag2 = projMgr.diag2.astype(np.uint8)
        imgFtr.drawHorizon(diag2)
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_COMPLEX
        if roadMgr.roadStraight:
            cv2.putText(middlepanel, 'Estimated lane curvature: road nearly straight',
                        (30, 60), font, 1, color, 2)
        elif roadMgr.radiusOfCurvature > 0.0:
            cv2.putText(middlepanel, 'Estimated lane curvature: center is %fm to the right' % (
                roadMgr.radiusOfCurvature), (30, 60), font, 1, color, 2)
        else:
            cv2.putText(middlepanel, 'Estimated lane curvature: center is %fm to the left' %
                        (-roadMgr.radiusOfCurvature), (30, 60), font, 1, color, 2)

        if roadMgr.lineBasePos < 0.0:
            cv2.putText(middlepanel, 'Estimated left of center: %5.2fcm' %
                        (-roadMgr.lineBasePos*1000), (30, 90), font, 1, color, 2)
        elif roadMgr.lineBasePos > 0.0:
            cv2.putText(middlepanel, 'Estimated right of center: %5.2fcm' % (
                roadMgr.lineBasePos*1000), (30, 90), font, 1, color, 2)
        else:
            cv2.putText(middlepanel, 'Estimated at center of road',
                        (30, 90), font, 1, color, 2)

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:720, 0:1280] = roadMgr.diag1

        # image filters
        diagScreen[0:240, 1280:1600] = cv2.resize(
            imgFtr.diag1, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[0:240, 1600:1920] = cv2.resize(
            imgFtr.diag2, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1280:1600] = cv2.resize(
            imgFtr.diag3, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1600:1920] = cv2.resize(
            imgFtr.diag4, (320, 240), interpolation=cv2.INTER_AREA)*4

        diagScreen[600:1080, 1280:1920] = cv2.resize(
            roadMgr.final, (640, 480), interpolation=cv2.INTER_AREA)

        diagScreen[720:840, 0:1280] = middlepanel

        # projection
        diagScreen[840:1080, 0:320] = cv2.resize(
            diag2, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 320:640] = cv2.resize(
            projMgr.diag1, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 640:960] = cv2.resize(
            projMgr.diag3, (320, 240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 960:1280] = cv2.resize(
            projMgr.diag4, (320, 240), interpolation=cv2.INTER_AREA)

        return diagScreen

    ########################################################
    # Diagnostics of the Projection Manager (Single)
    ########################################################
    def projectionHD(self):
        projMgr = self.pMgr
        imgFtr = self.rMgr.currentImageFtr

        # assemble the screen
        diagScreen = projMgr.diag3.astype(np.uint8)

        return diagScreen

    ########################################################
    # Diagnostics of the Projection Manager
    ########################################################
    def projectionDiag(self):
        projMgr = self.pMgr
        imgFtr = self.rMgr.currentImageFtr

        diag2 = projMgr.diag2.astype(np.uint8)
        imgFtr.drawHorizon(diag2)

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:540, 0:960] = cv2.resize(
            diag2, (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[0:540, 960:1920] = cv2.resize(projMgr.diag1.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 0:960] = cv2.resize(projMgr.diag3.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 960:1920] = cv2.resize(projMgr.diag4.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)

        return diagScreen

    ########################################################
    # Diagnostics of the Image Filters
    ########################################################
    def filterDiag(self):
        imgFtr = self.rMgr.currentImageFtr

        # assemble the screen
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:540, 0:960] = cv2.resize(
            imgFtr.diag1, (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[0:540, 960:1920] = cv2.resize(imgFtr.diag2.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 0:960] = cv2.resize(imgFtr.diag3.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)
        diagScreen[540:1080, 960:1920] = cv2.resize(imgFtr.diag4.astype(
            np.uint8), (960, 540), interpolation=cv2.INTER_AREA)

        return diagScreen
