import matplotlib;matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
import image_processing


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

fer_graph = {}
fer_graph_lock = threading.Lock()
fer_graph_event = threading.Event()

model = neural_nets.VggVdFaceFerDag()
model.eval()


def mult_255(img):
    return img * 255


vgg_transform = transforms.Compose(
                [transforms.Resize((model.meta["imageSize"][0], model.meta["imageSize"][1])),
                transforms.ToTensor(),
                transforms.Lambda(mult_255),
                transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])])

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

app = Flask(__name__)

old_expression = None
fig = plt.figure()
ax1 = plt.axes(ylim=(0,1))
line, = ax1.plot([])
plt.ylabel('Probability')
plotColors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plotLabels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
lines = []
for index in range(6):
    lobj = ax1.plot([],c=plotColors[index], label=plotLabels[index])[0]
    lines.append(lobj)

def fer_graph_processor():
    global fer_processing, fer_processing_lock, fer_processing_event, fer_graph, fer_graph_lock, fer_graph_event

    while True:
        fer_processing_event.wait()
        fer_processing_event.clear()

        with fer_processing_lock:
            expres_pdist = fer_processing['exp_pdist']
        # Update graph with new expres_pdist data points
        lines[0].set_ydata(np.append(lines[0].get_ydata, expres_pdist[0]))
        lines[1].set_ydata(np.append(lines[1].get_ydata, expres_pdist[1]))
        lines[2].set_ydata(np.append(lines[2].get_ydata, expres_pdist[2]))
        lines[3].set_ydata(np.append(lines[3].get_ydata, expres_pdist[3]))
        lines[4].set_ydata(np.append(lines[4].get_ydata, expres_pdist[4]))
        lines[5].set_ydata(np.append(lines[5].get_ydata, expres_pdist[5]))
        lines[6].set_ydata(np.append(lines[6].get_ydata, expres_pdist[6]))
        plt.show()

        # Save plot as numpy image
        img = None

        # Send the plot to fer_graph_generator
        with fer_graph_lock:
            fer_graph['plot'] = img
        fer_graph_event.set()


def fer_processor(print_interval=-1, log=None):
    global image, image_lock, image_event, fer_processing, fer_processing_lock, fer_processing_event

    fps_tracker = utils.FpsTracker(print_interval=print_interval, log=log)
    while True:
        # Wait for a new image to process from video_feed
        image_event.wait()
        image_event.clear()
        fps_tracker.track()

        with image_lock:
            # Image data is sent as string from FlaskSend.py
            np_img = np.frombuffer(image['data'], np.uint8)
            cropped = image['cropped']

        # Process the image and make it available for the fer_generator
        np_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if not cropped:
            face = image_processing.crop_face_transform(np_img)
            pil_img = Image.fromarray(face['img'])
            transformed_img = vgg_transform(pil_img)
            expression, exp_pdist = utils.get_expression(model, transformed_img, need_softmax=True)
        else:
            face = None
            pil_img = Image.fromarray(np_img)
            transformed_img = vgg_transform(pil_img)
            expression, exp_pdist = utils.get_expression(model, transformed_img, need_softmax=True)

        with fer_processing_lock:
            # Update fer_processing with the new information for fer_generator to use
            fer_processing['exp'] = expression
            fer_processing['exp_pdist'] = exp_pdist
            fer_processing['face'] = face
        # Let fer_generator know of the new data
        fer_processing_event.set()

        fps_tracker.frame_sent()

#old_expression = None
#fig = plt.figure()
#ax1 = plt.axes(ylim=(0,1))
#line, = plt.plot([])
#plt.ylabel('Probability')
#plotColors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#plotLabels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
#lines = []
#for index in range(6):
#    lobj = ax1.plot([],c=plotColors[index], label=plotLabels[index])[0]
#    lines.append(lobj)
#y0 = []
#y1 = []
#y2 = []
#y3 = []
#y4 = []
#y5 = []
#y6 = []

#def animate(i):
#    global old_expression;
#    expres_pdist = fer_processing['exp_pdist']
#    if(expres_pdist[0] != old_expression[0] or expres_pdist[1] != old_expression[1] or  expres_pdist[2] != old_expression[2] or expres_pdist[3] != old_expression[3] or expres_pdist[4] != old_expression[4] or expres_pdist[5] != old_expression[5] or expres_pdist[6] != old_expression[6] or expres_pdist[7] != old_expression[7]):
#        old_expression = expres_pdist
#        y0.append(expres_pdist[0])
#        y1.append(expres_pdist[1])
#        y2.append(expres_pdist[2])
#        y3.append(expres_pdist[3])
#        y4.append(expres_pdist[4])
#        y5.append(expres_pdist[5])
#        y6.append(expres_pdist[6])
#        ylist=[y0, y1, y2, y3, y4, y5, y6]

#        for lnum, line in enumerate(lines):
#            line.set_ydata(ylist[lnum])  # set data for each line separately.
#        return lines

def fer_graph_generator():
    global fer_graph, fer_graph_lock, fer_graph_event

    while True:
        fer_graph_event.wait()
        fer_graph_event.clear()

        with fer_graph_lock:
            plot = fer_graph['plot']

        success, plot = cv2.imencode(".jpg", plot, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if success:
            # yield the output frame in the byte format
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(plot) + b'\r\n'


def fer_generator():
    global image, image_lock

    fer_data = None

    # Continuously loop over the frames received from /video_feed
    while True:
        # sleep until a new processed image comes
        image_event.wait()
        image_event.clear()

        with image_lock:
            # Image data is sent as string from FlaskSend.py
            img = np.frombuffer(image['data'], np.uint8)
            cropped = image['cropped']

        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # Acquire lock for fer processing information and get the information
        if fer_processing_event.is_set():
            with fer_processing_lock:
                fer_data = fer_processing
            fer_processing_event.clear()
            print(f"Expression: {fer_data['exp']}\tProbabilities: {fer_data['exp_pdist']}")

        if fer_data is not None:
            if not cropped:
                face = fer_data['face']
                x, y = face['coord']
                w, h = face['size']
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                img = cv2.putText(img, fer_data['exp'], (max(x, 10), max(y-5, 10)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, str(fer_data['exp']), (10, 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        img = cv2.resize(img, (args.size[0], args.size[1]))

        success, img = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if success:
            # yield the output frame in the byte format
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n'


@app.route("/", methods=['GET'])
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/graph_feed", methods=['GET'])
def graph_feed():
    return Response(fer_graph_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed", methods=['GET'])
def video_feed():
    return Response(fer_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/cropped=<cropped>&width=<width>&height=<height>", methods=['POST'])
def video_feed_receive(cropped, width, height):
    global image, image_lock
    with image_lock:
        image['data'] = request.get_data()
        image['cropped'] = True if cropped == "True" else False
        image['size'] = (int(width), int(height))
    # Wakeup the image generator so it can process the new image
    image_event.set()
    response = json.dumps({'message': 'image received'})
    return Response(response=response, status=200, mimetype="application/json")


parser = argparse.ArgumentParser(description='Host server to receive images')
parser.add_argument('--host', dest='host', default=None, help="Web address to host the server")
parser.add_argument('--port', dest='port', default=None, help="Port to host the server")
parser.add_argument('--debug', dest='debug', action='store_true', help='Set server to debug mode')
parser.add_argument('--print', dest='print', type=int, default=-1, help='Set the print interval for FER FPS statistics')
parser.add_argument('--log', dest='log', default=None, help='File to log raw FPS data to')
parser.add_argument('--size', dest='size', nargs=2, type=int, default=[720, 480], metavar=('width', 'height'),
                    help='Specify what size image to show on the front-end (default: 720 x 480)')
args = parser.parse_args()


if __name__ == '__main__':
    # Start the thread that will be doing the facial expression recognition on the received images
    fer_processing_thread = threading.Thread(target=fer_processor, args=(args.print, args.log), daemon=True)
    fer_processing_thread.start()

    fer_graphing_thread = threading.Thread(target=fer_graph_processor, daemon=True)
    fer_graphing_thread.start()

    app.run(host=args.host, port=args.port, debug=args.debug)
