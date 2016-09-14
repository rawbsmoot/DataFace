#!/usr/bin/env python2


import time

start = time.time()

import argparse
import cv2
import os
import pickle
from datetime import datetime
import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    # print reps
    return reps


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)  # le - label and clf - classifer

    reps = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int, default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--highthresh', type=float, default=0.830)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str)

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    #count = 0
    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        persons, confidences = infer(frame, args)
        print "P: " + str(persons) + " C: " + str(confidences)
        if persons == ['robbie_smoot']:
            persons = "Rob"
        elif persons == ['john_marin']:
            persons = "Professor John"
        elif persons == ['mike_frantz']:
            persons = "Dr. Frantz"
        elif persons == ['lucy_smoot']:
            persons = "Lucy"
        elif persons == ['mike_ludwig']:
            persons = "Mike"
        elif persons == ['paul_trichon']:
            persons = "Paul"
        elif persons == ['leslie_pham']:
            persons = "Leslie"
        elif persons == ['ryan_gin']:
            persons = "Ryan"
        elif persons == ['roshanak_omrani']:
            persons = "Roshanak"
        elif persons == ['colin_clemence']:
            persons = "Colin"
        elif persons == ['pauline_chow']:
            persons = "Mrs. Chow"
        try:
            confidenceList.append('%.1f' % confidences[0])
        except:
            pass

        for i, c in enumerate(confidences):
            if c >= args.highthresh:
                cv2.putText(frame, "Hello {}!".format(persons),
                            (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 237, 88), 2)
        
                #needs work but somehow I'd like to be able to put new people in storage for labeling.
                #cv2.imwrite("/Users/smoot/Desktop/unknowns/unknown_%d.jpg" % count, frame)
                #count += 1
            elif c <= args.threshold:
                cv2.putText(frame, "Do I know you?".format(persons),
                            (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
            else:
                cv2.putText(frame, "{}, hold still pls..".format(persons),
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.putText(frame, "Confidence: {}".format(confidences),
                    (10, frame.shape[0] - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (5, 255, 220), 1)
                    
        cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (427, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    
        cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
