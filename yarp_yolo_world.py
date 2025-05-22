import sys
import numpy as np
import math
import string
import cv2
from collections import Counter

from ultralytics import YOLO
import os
import yarp
import torch
import threading


class yoloWDet(yarp.RFModule):
 
    def configure(self, rf) :
 
        self.period = 0.35
 
        self.model = YOLO('yolov8s-world.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
 
        self.yoloWDet_input_img_portName = '/yoloWDet/image:i' 
        self.yoloWDet_input_img_port = yarp.BufferedPortImageRgb()
        self.yoloWDet_input_img_port.open(self.yoloWDet_input_img_portName)
        self.in_buf_array = np.ones((480, 640, 3), dtype=np.uint8)
        self.in_buf_image = yarp.ImageRgb()
        self.in_buf_image.resize(640, 480)
        self.in_buf_image.setExternal(self.in_buf_array.data, self.in_buf_array.shape[1], self.in_buf_array.shape[0])
        self.image = np.ones((480, 640, 3), dtype=np.uint8)
 
        self.yoloWDet_dets_portName = '/yoloWDet/dets:o'
        self.yoloWDet_dets_port = yarp.BufferedPortBottle()
        self.yoloWDet_dets_port.open(self.yoloWDet_dets_portName)

        self.yoloWDet_img_out_portName = '/yoloWDet/image:o'
        self.yoloWDet_img_out_port = yarp.BufferedPortImageRgb()
        self.yoloWDet_img_out_port.open(self.yoloWDet_img_out_portName)
 
        self.yoloWDet_rpc_portName = "/yoloWDet/rpc:i"
        self.yoloWDet_rpc_port = yarp.Port()
        self.yoloWDet_rpc_port.open(self.yoloWDet_rpc_portName)
        self.attach(self.yoloWDet_rpc_port)

        self.confidence_threshold = 0.25

        self.current_classes = ['parrot', 'dog']
        self.model.set_classes(self.current_classes)

        self.lock = threading.Lock()

        self.label_colors = {}
 
        return True
 

    def update_classes(self, classes):
        if classes != self.current_classes:
            print(f"Updating classes from {self.current_classes} to {classes}")
            self.current_classes = classes
            self.model.set_classes(classes)
            self.label_colors = {label: (int(255 * (i / len(classes))), int(127 * (1 - i / len(classes))), int(255 * ((i + 1) % len(classes) / len(classes)))) for i, label in enumerate(classes)}
        return True

 
    def respond(self, command, reply):
        if command.get(0).asString()=='get_bbox':
            print('Received command GET_BBOX')

            objects_to_look = []
            for i in range(1, command.size()):
                obj = command.get(i).asString()
                objects_to_look.append(obj)

            print(f"Looking for: {objects_to_look}")
            
            with self.lock:
                self.update_classes(objects_to_look)
            
            reply.addString('Object list updated')
        elif command.get(0).asString() == 'quit':
            print('Received command QUIT')
            self.close()
            reply.addString('Quit command sent')
        return True
 

    def description_inference(self):

        description_btl_list = yarp.Bottle()
        received_image = self.yoloWDet_input_img_port.read()

        self.in_buf_image.copy(received_image)
        self.image = np.copy(self.in_buf_array)

        results = self.model.predict(
            self.image,
            conf=self.confidence_threshold
        )

        print('--- Detected Boxes ---')
        output_bottle = self.yoloWDet_dets_port.prepare()
        output_bottle.clear()

        out_image = self.image.copy()

        for result in results:
            bboxes = result.boxes.xyxy
            labels = result.boxes.cls
            scores = result.boxes.conf
            names = result.names

            for i, box in enumerate(bboxes):
                object_bottle = output_bottle.addList()
                bbox_list = object_bottle.addList()
                for value in box.tolist():
                    bbox_list.addFloat64(value)

                centroid_list = object_bottle.addList()
                centroid_list.addInt64(int((bbox_list.get(0).asFloat64()+bbox_list.get(2).asFloat64())/2))
                centroid_list.addInt64(int((bbox_list.get(1).asFloat64()+bbox_list.get(3).asFloat64())/2))
                label_index = int(labels[i].item())
                label_name = names[label_index]
                confidence = float(scores[i].item())

                object_bottle.addString(label_name)
                object_bottle.addFloat64(confidence)

                x1, y1, x2, y2 = map(int, box.tolist())
                color = tuple(min(255, int(c * 1.5)) for c in self.label_colors.get(label_name, (0, 255, 0)))
                cv2.rectangle(out_image, (x1, y1), (x2, y2), color, thickness=4)
                cv2.putText(out_image, f"{label_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thickness=3, lineType=cv2.LINE_AA)

        out_buf_array = out_image.astype(np.uint8)
        out_buf_image = self.yoloWDet_img_out_port.prepare()
        out_buf_image.resize(out_image.shape[1], out_image.shape[0])
        out_buf_image.setExternal(out_buf_array.data, out_buf_array.shape[1], out_buf_array.shape[0])
        self.yoloWDet_img_out_port.write()

        self.yoloWDet_dets_port.write()



        return True


    def updateModule(self):
 
        print('Running')

        with self.lock:
            self.description_inference()

        return True
 
 
    def getPeriod(self):
        return self.period
    
        
    def close(self):
        self.yoloWDet_input_img_port.close()       
        self.yoloWDet_dets_port.close()
        self.yoloWDet_img_out_port.close()        
        return True
 
 
    def interruptModule(self):
        self.yoloWDet_input_img_port.interrupt()    
        self.yoloWDet_dets_port.interrupt() 
        self.yoloWDet_img_out_port.interrupt()      
        return True
    
#########################################
 
if __name__ == '__main__':
    
    yarp.Network.init()
 
    mod = yoloWDet()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
    yarp.Network.fini()
