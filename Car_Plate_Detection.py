import cv2
import numpy as np
class Car_Plate_Detection:
    def __init__(self):
        modelConfiguration = "CarPlateModel/yolov3-tiny.cfg"
        modelWeights = "CarPlateModel/yolov3-tiny.backup"
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    def getOutputsNames(self, n):
        layersNames = n.getLayerNames()
        return [layersNames[i[0] - 1] for i in n.getUnconnectedOutLayers()]

    def postprocess(self, fr, outs, confT, nmsT):
        frameHeight = fr.shape[0]
        frameWidth = fr.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for o in outs:
            for detection in o:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confT:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confT, nmsT)
        cropped = None
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = max(box[0], 0)
            top = max(box[1], 0)
            width = max(box[2], 0)
            height = max(box[3], 0)
            if height > width:
                continue
            cropped = fr[top:(top + height), left:(left + width)]

        return len(indices) > 0, cropped
    def Detect_Plate(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        run = self.net.forward(self.getOutputsNames(self.net))
        rec, plateImg = self.postprocess(frame, run, 0.5, 0.5)
        x = np.array([2])
        if rec > 0 and type(plateImg) == type(x) and plateImg.all() is not None:
            cv2.imwrite("Plates From Model/0.png", plateImg.astype(np.uint8))
            return plateImg

        return False