from flask import Flask, Response, render_template, request
from PIL import Image
import argparse
import threading
import json
import cv2
import numpy as np
import torch
import utils

import neural_nets
import torchvision.transforms as transforms

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
image = None
image_lock = threading.Lock()
app = Flask(__name__)

model = neural_nets.VggVdFaceFerDag()
model.eval()
vgg_transform = transforms.Compose(
                [transforms.Resize((model.meta["imageSize"][0], model.meta["imageSize"][1])),
                transforms.ToTensor(),
                lambda x: x * 255,
                transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])])
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

@app.route("/", methods=['GET'])
def index():
    # return the rendered template
    return render_template("index.html")


def generate():
    # grab global references to the output frame and lock variables
    global image, image_lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with image_lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if image is None:
                continue

            # convert string of image data to uint8
            nparr = np.fromstring(image, np.uint8)
            image = None

        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_img = Image.fromarray(img)
        t_img = vgg_transform(pil_img)
        expression = utils.get_expression(model, t_img)
        print(expression)
        flag, img = cv2.imencode(".jpg", img)
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(img) + b'\r\n')


@app.route("/video_feed", methods=['GET', 'POST'])
def video_feed():
    global image, image_lock
    if request.method == 'GET':
        return Response(generate(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    elif request.method == 'POST':
        with image_lock:
            image = request.get_data()
        response = json.dumps({'message': 'image received'})
        return Response(response=response, status=200, mimetype="application/json")
    response = json.dumps({'message': 'invalid usage'})
    return Response(response=response, status=400, mimetype="application/json")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Host server to receive images')
    parser.add_argument('--host', dest='host', default=None, help="Web address to host the server")
    parser.add_argument('--port', dest='port', default=None, help="Port to host the server")
    parser.add_argument('--debug', dest='debug', action='store_true', help='Set server to debug mode')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
