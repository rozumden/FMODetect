import argparse
import numpy as np
import tensorflow as tf
from net_model import *
import imageio
import os
import cv2
import skimage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False, default=None)
    parser.add_argument("--model", required=False, default="FMODetect.h5")
    parser.add_argument("--save", required=False, default="example")
    parser.add_argument("--median", required=False, default=3)
    parser.add_argument("--average", required=False, default=True)
    return parser.parse_args()

def interpolate_fifa(im):
    im0 = im.copy()
    im0[1:-1:2] = (im0[:-2:2] + im0[2::2])/2
    im1 = im.copy()
    im1[2:-2:2] = (im1[1:-3:2] + im1[3:-1:2])/2
    return im0, im1

def get_frame(frame, inc_res = 2):
    sh = frame.shape[:2]
    sh0 = int(sh[0]/6)
    sh1 = int(sh[1]/6)
    # frame_crop = frame
    frame_crop = frame[2*sh0:-3*sh0,2*sh1:-3*sh1]
    frame_crop = skimage.transform.resize(frame_crop, (frame_crop.shape[0]*inc_res, frame_crop.shape[1]*inc_res), order=3)
    return frame_crop

def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = tf.keras.models.load_model(args.model, custom_objects={ 'fmo_loss_function': custom_loss(None) })

    ## estimate initial background 
    Ims = []
    cap = cv2.VideoCapture(args.video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not frame is None:
            frame = frame / 255
            frame0, frame1 = interpolate_fifa(frame)
            if args.average:
                Ims.append(get_frame((frame0 + frame1)/2))
            else:
                Ims.append(get_frame(frame0))
                if len(Ims) < args.median:
                    Ims.append(get_frame(frame1))

        if len(Ims) >= args.median:
            break
    B = np.median(np.asarray(Ims)/255, 0)[:,:,[2,1,0]]

    ## run FMODetect
    shape = process_image(B).shape
    out = cv2.VideoWriter(os.path.join(args.save, 'detections.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shape[1], shape[0]),True)
    frmi = 0
    frame1 = None
    while cap.isOpened():
        if frmi < args.median:
            frame = Ims[frmi]
        else:
            if frame1 is None:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame / 255
                frame0, frame1 = interpolate_fifa(frame)
                if args.average:
                    frame = get_frame( (frame0 + frame1)/2 )
                    frame1 = None
                else:
                    frame = get_frame(frame0)
            else:
                frame = get_frame(frame1)
                frame1 = None
            Ims = Ims[1:]
            Ims.append(frame)
            ## update background (running median)
            B = np.median(np.asarray(Ims), 0)[:,:,[2,1,0]]
        frmi += 1
        if args.average:
            mult = 1
        else:
            mult = 2
        if frmi < mult*88:
            continue

        I = process_image(frame[:,:,[2,1,0]])
        X = np.concatenate((I,process_image(B)),2)[None]
        predictions = model.predict(X)
        predictions[predictions < 0] = 0
        predictions[predictions > 1] = 1
        Io = I - I.min()
        Io = Io / Io.max()

        imageio.imwrite('tmpi.png',frame[:,:,[2,1,0]])
        imageio.imwrite('tmpb.png',B)
        imageio.imwrite('tmpo.png',predictions[0][:,:,[0,0,0]])
        breakpoint()

        # out.write( (predictions[0][:,:,[0,0,0]] * 255).astype(np.uint8) )
        out.write( (predictions[0][:,:,[0,0,0]]*Io[:,:,[2,1,0]]).astype(np.uint8) )

    cap.release()
    out.release()
    
if __name__ == "__main__":
    main()