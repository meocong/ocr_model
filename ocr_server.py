from flask import Flask, request, jsonify
app = Flask(__name__)
import io
import os
from PIL import Image
import numpy as np
from random import randint
import time
import cv2
import json

import evaluate_model
from model.utils.image import greyscale

models = evaluate_model.init_model(model_saved_path='./results2/full/', vocab_path="./data2/vocab.txt")
print("Loaded model")

def read_image(img_bytes):
    # if (img_bytes is np.array):
    #     return img_bytes
    return cv2.imdecode(np.asarray(bytearray(img_bytes.read()), dtype="uint8"), cv2.IMREAD_GRAYSCALE)

@app.route("/", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    start_time = time.time()
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image", None) is not None:
            try:
                # Read image by Opencv
                image = read_image(request.files["image"])
                image = greyscale(image)

                # Read image by PIL
                # image = Image.open(io.BytesIO(request.files["image"].read()))

                # classify the input image
                temp = models.predict_batch(images=[image])

                data = {}
                data["prediction"] = temp[0][0]
                #data["confidence"] = temp[1][0]
                # indicate that the request was a success
                data["success"] = True
            except Exception as ex:
                data['error'] = str(ex)
                # print(ex)
        elif request.files.getlist("images"):
            try:
                # read the image in PIL format
                # images = [Image.open(io.BytesIO(x.read())) for x in request.files.getlist("images")]

                # Read images in Opencv format
                images = [read_image(x) for x in request.files.getlist("images")]
                images = [greyscale(image) for image in images]

                temp = models.predict_batch(images=[images])

                data = {}
                data["prediction"] = temp[0]
                #data["confidence"] = temp[1]
                # indicate that the request was a success
                data["success"] = True
            except Exception as ex:
                data['error'] = str(ex)
                # print(ex)
        elif request.data is not None:
            try:
                # Read image by Opencv
                image = cv2.imdecode(np.fromstring(request.data, np.uint8), cv2.IMREAD_GRAYSCALE)

                # Read image by PIL
                # image = Image.open(io.BytesIO(request.files["image"].read()))

                # classify the input image
                temp = models.predict_batch(images=[image])

                data = {}
                data["prediction"] = temp[0][0]
                data["confidence"] = temp[1][0]
                # indicate that the request was a success
                data["success"] = True
            except Exception as ex:
                data['error'] = str(ex)
            #     print(ex)

    data['run_time'] = "%.2f" % (time.time() - start_time)
    # return the data dictionary as a JSON response
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)