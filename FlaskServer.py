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
from image_processing import crop_face_transform


# Contains the unprocessed image from video_feed
image = {}
# This lock allows for the image to be exchanged between the video_feed endpoint and fer_processor
image_lock = threading.Lock()
# Event signaling the receipt of a new image in video_feed
image_event = threading.Event()

# Contains the processed image and expression from fer_processor
fer_processing = {}
# This lock allows for the decoded image and expression to be exchanged between fer_processor and fer_generator
fer_processing_lock = threading.Lock()
# Event signaling the processing of a new image
fer_processing_event = threading.Event()

model = neural_nets.VggVdFaceFerDag()
model.eval()
vgg_transform = transforms.Compose(
                [transforms.Resize((model.meta["imageSize"][0], model.meta["imageSize"][1])),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255),
                transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])])

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    # return the rendered template
    return render_template("index.html")


def fer_processor():
    global image, image_lock, image_event, fer_processing, fer_processing_lock, fer_processing_event

    while True:
        # Wait for a new image to process from video_feed
        image_event.wait()
        image_event.clear()

        with image_lock:
            # Image data is sent as string from FlaskSend.py
            np_img = np.frombuffer(image['data'], np.uint8)
            cropped = image['cropped']

        # Process the image and make it available for the fer_generator
        np_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if not cropped:
            np_img = crop_face_transform(np_img)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        pil_img = Image.fromarray(np_img)
        transformed_img = vgg_transform(pil_img)
        expression, exp_pdist = utils.get_expression(model, transformed_img, need_softmax=True)
        cv2.putText(np_img, str(expression), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        success, img = cv2.imencode(".jpg", np_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if success:
            with fer_processing_lock:
                # Update fer_processing with the new information for fer_generator to use
                fer_processing['img'] = img
                fer_processing['exp'] = expression
                fer_processing['exp_pdist'] = exp_pdist
            # Let fer_generator know of the new data
            fer_processing_event.set()


def fer_generator():
    global image, image_lock

    # Continuously loop over the frames received from /video_feed
    while True:
        # sleep until a new processed image comes
        fer_processing_event.wait()
        fer_processing_event.clear()

        # Acquire lock for fer processing information and get the information
        with fer_processing_lock:
            img = fer_processing['img']
            expression = fer_processing['exp']
            expression_pdist = fer_processing['exp_pdist']

        print(f"Expression: {expression}")
        print(f"Expression probability distribution: {expression_pdist}")

        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n'


@app.route("/video_feed", methods=['GET'])
def video_feed():
    return Response(fer_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/cropped=<cropped>", methods=['POST'])
def video_feed_receive(cropped):
    global image, image_lock
    with image_lock:
        image['data'] = request.get_data()
        image['cropped'] = True if cropped == "True" else False
    # Wakeup the image generator so it can process the new image
    image_event.set()
    response = json.dumps({'message': 'image received'})
    return Response(response=response, status=200, mimetype="application/json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Host server to receive images')
    parser.add_argument('--host', dest='host', default=None, help="Web address to host the server")
    parser.add_argument('--port', dest='port', default=None, help="Port to host the server")
    parser.add_argument('--debug', dest='debug', action='store_true', help='Set server to debug mode')
    args = parser.parse_args()

    # Start the thread that will be doing the facial expression recognition on the received images
    fer_processing_thread = threading.Thread(target=fer_processor, daemon=True)
    fer_processing_thread.start()

    app.run(host=args.host, port=args.port, debug=args.debug)
