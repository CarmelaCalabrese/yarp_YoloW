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
 
class yoloWDet(yarp.RFModule):
 
    def configure(self, rf) :
 
        self.period = 4
 
        self.model = YOLO('yolov8s-world.pt')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # # Also try forcing CLIP submodules to CUDA
        # if hasattr(self.model.model, "text_model"):
        #     self.model.model.text_model.to(self.device)
 
        self.yoloWDet_input_img_portName = '/yoloWDet/image:i' 
        self.yoloWDet_input_img_port = yarp.BufferedPortImageRgb()
        self.yoloWDet_input_img_port.open(self.yoloWDet_input_img_portName)
        self.in_buf_array = np.ones((480, 640, 3), dtype=np.uint8)
        self.in_buf_image = yarp.ImageRgb()
        self.in_buf_image.resize(640, 480)
        self.in_buf_image.setExternal(self.in_buf_array.data, self.in_buf_array.shape[1], self.in_buf_array.shape[0])
        self.image = np.ones((480, 640, 3), dtype=np.uint8)
 
        self.yoloWDet_dets_portName = '/yoloWDet/dets:o' #detections from /detection/dets:o
        self.yoloWDet_dets_port = yarp.BufferedPortBottle()
        self.yoloWDet_dets_port.open(self.yoloWDet_dets_portName)
 
        self.yoloWDet_rpc_portName = "/yoloWDet/rpc:i"
        self.yoloWDet_rpc_port = yarp.Port()
        self.yoloWDet_rpc_port.open(self.yoloWDet_rpc_portName)
        self.attach(self.yoloWDet_rpc_port)

        self.confidence_threshold = 0.25

        #self.current_classes = ['person', 'book', 'chair', 'dog']
        self.current_classes = ['parrot', 'dog']
 
        return True
 

    def update_classes(self, classes):
        if classes != self.current_classes:
            print(f"Updating classes from {self.current_classes} to {classes}")
            self.current_classes = classes
            self.model.set_classes(classes)
        return True

 
    def respond(self, command, reply):
        if command.get(0).asString()=='get_bbox':
            print('Received command GET_BBOX')
            obj2look=command.get(1).asString()
            self.update_classes(obj2look)
            # obj2look = ['chair', 'book']
            # self.description_inference(obj2look)
            reply.addString('Ciao')
            #reply.addString(self.inf_description)
        elif command.get(0).asString() == 'quit':
            print('Received command QUIT')
            self.close()
            reply.addString('Quit command sent')
        return True
 

 
    def description_inference(self, classes):

        print(self.current_classes)

        description_btl_list = yarp.Bottle()
        received_image = self.yoloWDet_input_img_port.read()

        self.in_buf_image.copy(received_image)
        self.image = np.copy(self.in_buf_array)

        # # Convert to torch tensor, normalize, change shape to CHW, add batch
        # img_tensor = torch.from_numpy(self.image).permute(2, 0, 1).float() / 255.0  # [3, H, W], float32 in [0,1]
        # img_tensor = img_tensor.unsqueeze(0).to("cuda")  # [1, 3, H, W], on CUDA


        results = self.model.predict(
        self.image,
        conf=self.confidence_threshold,
        device = 'cuda:0'
        )

        print('--- Detected Boxes ---')
        # Draw bounding boxes
        for result in results:
            bboxes = result.boxes.xyxy
            print(bboxes)
            for i, row in enumerate(bboxes):
                object_bottle = yarp.Bottle()
                print(f"Row {i} as list:", row.tolist())
                box = row.tolist()
                #label = det['label']
                #conf = det['score']
                
                # First: bounding box sublist
                bbox_list = object_bottle.addList()
                for value in box:
                    bbox_list.addFloat64(value)

                # Second: label
                object_bottle.addString('person')

                # Third: confidence score
                object_bottle.addFloat64(0.65)

                #object_bottle.addList(box) #bbox
                #object_bottle.addString(label) #description
                #object_bottle.addFloat64(score) #diag
                
                #description_btl_list.addList().read(object_bottle)
                            
        #return description_btl_list
        return True

    
    def updateModule(self):
 
        print('Running')

        # new_classes = ['cup', 'cat', 'person']

        # if new_classes:
        #     self.update_classes(new_classes)
        
        self.description_inference(self.current_classes)

        return True
 
 
    def getPeriod(self):
        return self.period
    
        
    def close(self):
        self.yoloWDet_input_img_port.close()       
        self.yoloWDet_dets_port.close()                  
        return True
 
 
    def interruptModule(self):
        self.yoloWDet_input_img_port.interrupt()    
        self.yoloWDet_dets_port.interrupt()       
        return True
    
#########################################
 
if __name__ == '__main__':
    
    yarp.Network.init()
 
    mod = yoloWDet()
    #mod.description_inference(['chair', 'book'])
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    mod.runModule(rf)
    yarp.Network.fini()