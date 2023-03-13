from flask import Flask, request, jsonify
from yolov4 import Create_Yolo #yolov3 only
from utils import detect_image
from configs import *
import cv2
import numpy as np
import base64
import sys

app = Flask(__name__)

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"yolov3_custom")

@app.route('/detect', methods=['POST'])
def detect():

    try:
        image = request.files['image'].read()
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("pred10.jpeg", image)
        
        
        image_path = "pred10.jpeg"
        image = cv2.imread(image_path)
        height, width , _= image.shape

        if width < height:
            rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(image_path, rotated_img)
        image,des,dis,pth,imgg=detect_image(yolo, image_path, input_size=YOLO_INPUT_SIZE, score_threshold=0.3, iou_threshold=0.45, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0),em=0)
        cv2.imwrite("output.jpeg", image)

        with open("output.jpeg", 'rb') as f:
            image_data = f.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')

        
        return jsonify({'img1':image_base64,'img2':imgg,'des':des,'dis':dis,'pth':pth})
    except Exception as e:
        exception_type, exception_object, exception_traceback = sys.exc_info()
        line_number = exception_traceback.tb_lineno
        return jsonify({'img1':"error",'img2':str(line_number)+str(exception_type)+" "+str(e),'des':"",'dis':"",'pth':""})

@app.route('/')
def frst():
    return "vignesh da..."
    

if __name__ == '__main__':
    app.run()
