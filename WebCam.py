import argparse
from PIL import Image
import cv2
import image_processing
import neural_nets as nns
import torchvision.transforms as transforms
import torch
import utils


parser = argparse.ArgumentParser(description='Perform Face Crop and/or FER from your web cam')
parser.add_argument('-j', '--jetson', dest='jetson', action='store_true',
                    help='Flag specifying if you are running this from an nvidia jetson')
parser.add_argument('--fer', dest='fer', action='store_true',
                    help='Flag specifying if you would like to run facial expression recognition')
parser.add_argument('-f', '--frames', dest='frames', default=-1, type=int, metavar='F',
                    help='Amount of frames to capture from the web cam')
args = parser.parse_args()

if args.fer:
    model = nns.VggVdFaceFerDag()
    transform = transforms.Compose(
                    [transforms.Resize(model.meta["imageSize"][0]),
                     transforms.ToTensor(),
                     transforms.Lambda(lambda x: x*255),
                     transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])])

    model.eval()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

if args.jetson:
    cap = cv2.VideoCapture(utils.jetson_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

if args.frames < 0:
    capture_forever = True
else:
    capture_forever = False

captured_frames = 0
while capture_forever or captured_frames < args.frames:
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("ERROR: Could not capture from web cam")
        break
    captured_frames += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = image_processing.crop_faces(rgb_frame, fx=0.2, fy=0.2)

    for f in faces:
        x, y = f["coord"]
        w, h = f["size"]
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if args.fer:
            expression, _ = utils.get_expression(model, transform(Image.fromarray(f["img"])), need_softmax=True)
            frame = cv2.putText(frame, expression, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
