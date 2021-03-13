import argparse
import numpy as np
import tensorflow as tf
from net_model import *
import imageio
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False, default=None)
    parser.add_argument("--im", required=False, default=None)
    parser.add_argument("--bgr", required=False, default=None)
    parser.add_argument("--model", required=False, default="FMODetect.h5")
    parser.add_argument("--save", required=False, default="example")
    parser.add_argument("--median", required=False, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = tf.keras.models.load_model(args.model, custom_objects={ 'fmo_loss_function': custom_loss(None) })
    
    if args.video is not None:
        ## estimate initial background 
        Ims = []
        cap = cv2.VideoCapture(args.video)
        while cap.isOpened():
            ret, frame = cap.read()
            Ims.append(frame)
            if len(Ims) >= args.median:
                break
        B = np.median(np.asarray(Ims)/255, 0)[:,:,[2,1,0]]

        ## run FMODetect
        shape = process_image(B).shape
        out = cv2.VideoWriter(os.path.join(args.save, 'detections.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shape[1], shape[0]),True)
        frmi = 0
        while cap.isOpened():
            if frmi < args.median:
                frame = Ims[frmi]
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                Ims = Ims[1:]
                Ims.append(frame)
                ## update background (running median)
                B = np.median(np.asarray(Ims)/255, 0)[:,:,[2,1,0]]
            frmi += 1
            I = process_image(frame[:,:,[2,1,0]]/255)
            X = np.concatenate((I,process_image(B)),2)[None]
            predictions = model.predict(X)
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1
            Io = I - I.min()
            Io = Io / Io.max()
            # out.write( (predictions[0][:,:,[0,0,0]] * 255).astype(np.uint8) )
            out.write( (predictions[0][:,:,[0,0,0]]*Io[:,:,[2,1,0]] * 255).astype(np.uint8) )

        cap.release()
        out.release()
    else:
        if args.im is None:
            ims = []
            bgrs = []
            for ss in range(2):
                ims.append(os.path.join('example','ex{:1d}_im.png'.format(ss)))
                bgrs.append(os.path.join('example','ex{:1d}_bgr.png'.format(ss)))
        else:
            ims = [args.im]
            bgrs = [args.bgr]

        for ss in range(len(ims)):
            I = process_image(get_im(ims[ss]))
            B = process_image(get_im(bgrs[ss]))
            X = np.concatenate((I,B),2)[None]

            predictions = model.predict(X)
            predictions[predictions < 0] = 0
            predictions[predictions > 1] = 1
            
            imageio.imwrite(os.path.join(args.save,"ex{:1d}_tdf.png".format(ss)), predictions[0])
            Io = I - I.min()
            Io = Io / Io.max()
            imageio.imwrite(os.path.join(args.save,"ex{:1d}_tdfim.png".format(ss)), predictions[0][:,:,[0,0,0]]*Io[:,:,[2,1,0]])

if __name__ == "__main__":
    main()