import random
from algorithm_framework import ROSCVAlgorithm, BoundingBox, BBVector
import numpy as np
import cv2
import gluoncv as gcv
import mxnet as mx
from mxnet import nd
from gluoncv.data.transforms import image as timage
import os 

class MyDetector(ROSCVAlgorithm):

    def __init__(self, executor):
        super(MyDetector,self).__init__()
        self.CLASSES = ["truck", "car", "van", "bus"]
        self.executor = executor

        
    def pre_process(self, frame):
        frame = cv2.resize(frame, (320, 320))
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        #img = timage.imresize(frame, 320, 320, interp=9)
        #orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(frame)
        img = mx.nd.image.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        tensor = img.expand_dims(0)
        return tensor  
    
    
    def processImage(self, callback, image):
        # In the real world, here we would process image and detect objects
        # Convert it to a numpy array and use OpenCV to turn it to RGB from BGR
        input_image = np.array(image)
        ori_height, ori_width, _ = input_image.shape
        h_scale = ori_height/320.
        w_scale = ori_width/320.
        tensor = self.pre_process(input_image)
        class_IDs, scores, bounding_boxes = self.executor.forward(is_train=False, data=tensor)
        class_IDs, scores, bounding_boxes = class_IDs[0].asnumpy(), scores[0].asnumpy(), bounding_boxes[0].asnumpy()
        
        # For the PoC, build a list of 5 dummy BoundingBoxes
        result = []
        for i in range(0, class_IDs.shape[0]):
            bb = BoundingBox(self.CLASSES[int(class_IDs[i])])
            bb.probability = float(scores[i])
            if bb.probability < 0.4:
                 break
            bb.x1, bb.y1, bb.x2, bb.y2 = [int((bounding_boxes[i, 0] * w_scale)), int((bounding_boxes[i, 1] * h_scale)),
                                  int((bounding_boxes[i, 2] * w_scale)), int((bounding_boxes[i, 3] * h_scale))]

            bb.alpha = 0
            if int(bb.x1) > 0 and int(bb.y1) > 0 and int(bb.x2) > 0 and int(bb.y2) > 0:
                result.append(bb)

        # Build a vector out of the list
        results = BBVector(result)
        # Execute the callback with our vector of BoundingBoxes as argument
        self.result_function(callback, results)

def main():
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'  #use mxnet autotune cudnn for about 2x speed up 
    #load model and params
    sym, arg_params, aux_params = mx.model.load_checkpoint('models/mobilenet0.25_yolo3_final', 0)
    executor = sym.simple_bind(ctx=mx.gpu(0), data=(1,3,320,320), grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)
    #warm up for search the best config in cudnn
    print("warm up for cudnn config......")
    a = executor.forward(is_train=False, data=mx.nd.zeros((1,3,320,320)))
    nd.waitall()
    print("start!")
    
    cva = MyDetector(executor)
    cva.Run()

if __name__ == "__main__":
    main()
