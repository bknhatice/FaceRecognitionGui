import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import cv2

import os

import dlib

import argparse

import pickle
import math

import threading
import logging

from sklearn.lda import LDA

from sklearn.mixture import GMM
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import time

from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
import atexit
from subprocess import Popen, PIPE
import os.path
import numpy as np
import pandas as pd

import openface
import aligndlib
logger = logging.getLogger(__name__)

start = time.time()
np.set_printoptions(precision=2)

fileDir = os.path.dirname(os.path.realpath(__file__))

luaDir = "/home/hatice/PycharmProjects/newOpenface/batch-represent"
modelDir = "/home/hatice/PycharmProjects/newOpenface/models"
dlibModelDir = "/home/hatice/PycharmProjects/newOpenface/models/dlib"
openfaceModelDir =  "/home/hatice/PycharmProjects/newOpenface/models/openface"

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default='/home/hatice/PycharmProjects/newOpenface/models/openface/nn4.small2.v1.t7')
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
args = parser.parse_args()

class FaceRecogniser(object):
    """This class implements face recognition using Openface's
    pretrained neural network and a Linear SVM classifier. Functions
    below allow a user to retrain the classifier and make predictions
    on detected faces"""
    def __init__(self):
        self.net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,cuda=args.cuda)
        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.neuralNetLock = threading.Lock()
        self.predictor = dlib.shape_predictor(args.dlibFacePredictor)

        print("Opening classifier.pkl to load existing known faces db")
        with open("generated-embeddings/classifier.pkl", 'r') as f: # le = labels, clf = classifier
            (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM

    def make_prediction(self,rgbFrame,bb):
        """The function uses the location of a face
        to detect facial landmarks and perform an affine transform
        to align the eyes and nose to the correct positiion.
        The aligned face is passed through the neural net which
        generates 128 measurements which uniquly identify that face.
        These measurements are known as an embedding, and are used
        by the classifier to predict the identity of the person"""

        landmarks = self.align.findLandmarks(rgbFrame, bb)
        if landmarks == None:
            print("///  FACE LANDMARKS COULD NOT BE FOUND  ///")
            return None
        alignedFace = self.align.align(args.imgDim, rgbFrame, bb,landmarks=landmarks,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            print("///  FACE COULD NOT BE ALIGNED  ///")
            return None

        print("////  FACE ALIGNED  // ")
        with self.neuralNetLock :
            persondict = self.recognize_face(alignedFace)

        if persondict is None:
            print("/////  FACE COULD NOT BE RECOGNIZED  //")
            return persondict, alignedFace
        else:
            print("/////  FACE RECOGNIZED  /// ")
            return persondict, alignedFace

    def recognize_face(self,img):
        if self.getRep(img) is None:
            return None
        rep1 = self.getRep(img) # Gets embedding representation of image
        print (rep1)
        print("Embedding returned. Reshaping the image and flatting it out in a 1 dimension array.")
        rep = rep1.reshape(1, -1)   #take the image and  reshape the image array to a single line instead of 2 dimensionals
        start = time.time()
        print("Submitting array for prediction.")
        predictions = self.clf.predict_proba(rep).ravel() # Computes probabilities of possible outcomes for samples in classifier(clf).
        #print("We need to dig here to know why the probability are not right.")
        print predictions

        maxI = np.argmax(predictions)
        print maxI
        person1 = self.le.inverse_transform(maxI)
        print person1
        confidence1 = int(math.ceil(predictions[maxI]*100))
        print confidence1

        print("Recognition took {} seconds.".format(time.time() - start))
        print("Recognized {} with {:.2f} confidence.".format(person1, confidence1))

        persondict = {'name': person1, 'confidence': confidence1, 'rep':rep1}
        return persondict

    def getRep(self,alignedFace):
        bgrImg = alignedFace
        if bgrImg is None:
            print ("unable to load image")

        print("Tweaking the face color ")

        read = cv2.imread(bgrImg)
        dim = (96, 96)
        align = cv2.resize(read, dim, interpolation=cv2.INTER_AREA)

        alignedFace = cv2.cvtColor(align, cv2.COLOR_BGR2RGB)
        start = time.time()
        print("Getting embedding for the face")
        rep = self.net.forward(alignedFace) # Gets embedding - 128 measurements
        return rep


    def reloadClassifier(self):
         with open("generated-embeddings/classifier.pkl", 'r') as f: # Reloads character stream from pickle file
             (self.le, self.clf) = pickle.load(f) # Loads labels and classifier SVM or GMM
         print("reloadClassifier called")


         return True

    def trainClassifier(self):
        """Trainng the classifier begins by aligning any images in the
        img directory and putting them into the aligned images
        directory. Each of the aligned face images are passed through the
        neural net and the resultant embeddings along with their
        labels (names of the people) are used to train the classifier
        which is saved to a pickle file as a character stream"""

        print("trainClassifier called")

        path = "/home/hatice/PycharmProjects/newOpenface/mini/cache.t7"
        try:
            os.remove(path) # Remove cache from aligned images folder
        except:
            print("Failed to remove cache.t7. Could be that it did not existed in the first place.")
            pass

        start = time.time()
        OUTER_EYES_AND_NOSE = [36, 45, 33]
        aligndlib.alignMain("img/","mini/","outerEyesAndNose","/home/hatice/PycharmProjects/newOpenface/models/dlib/shape_predictor_68_face_landmarks.dat",args.imgDim)
        print("Aligning images for training took {} seconds.".format(time.time() - start))
        done = False
        start = time.time()

        done = self.generate_representation()

        if done is True:
            print("Representation Generation (Classification Model) took {} seconds.".format(time.time() - start))
            start = time.time()
            # Train Model
            self.train("/home/hatice/PycharmProjects/newOpenface/generated-embeddings/","LinearSvm",-1)
            print("Training took {} seconds.".format(time.time() - start))
        else:
            print("Generate representation did not return True")


    def generate_representation(self):
        print("lua Directory:    " + luaDir)
        self.cmd = ['/usr/bin/env', 'th', "/home/hatice/PycharmProjects/newOpenface/batch-represent/main.lua", '-outDir',  "/home/hatice/PycharmProjects/newOpenface/generated-embeddings/" , '-data', "/home/hatice/PycharmProjects/newOpenface/mini/"]
        print("lua command:    " + str(self.cmd))
        if args.cuda:
            self.cmd.append('-cuda')
            print("using -cuda")
        self.p = Popen(self.cmd, stdin=PIPE, stdout=PIPE, bufsize=0)
        #our issue is here, torch probably crashes without giving much explanation.
        outs, errs = self.p.communicate() # Wait for process to exit - wait for subprocess to finish writing to files: labels.csv & reps.csv
        print("Waiting for process to exit to finish writing labels and reps.csv" + str(outs) + " - " + str(errs))

        def exitHandler():
            if self.p.poll() is None:
                print("<=Something went Wrong===>")
                self.p.kill()
                return False
        atexit.register(exitHandler)

        return True

    def train(self,workDir,classifier,ldaDim):
        fname = "{}labels.csv".format(workDir) #labels of faces
        print("Loading labels " + fname + " csv size: " +  str(os.path.getsize("/home/hatice/PycharmProjects/newOpenface/generated-embeddings/reps.csv")))
        if os.path.getsize(fname) > 0:
            print(fname + " file is not empty")
            labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
            print(labels)
        else:
            print(fname + " file is empty")
            labels = "1:mini/dummy/1.png"  #creating a dummy string to start the process
        logger.debug(map(os.path.dirname, labels))
        logger.debug(map(os.path.split,map(os.path.dirname, labels)))
        logger.debug(map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels))))
        labels = map(itemgetter(1),map(os.path.split,map(os.path.dirname, labels)))

        fname = "{}reps.csv".format(workDir) # Representations of faces
        fnametest = format(workDir) + "reps.csv"
        print("Loading embedding " + fname + " csv size: " + str(os.path.getsize(fname)))
        if os.path.getsize(fname) > 0:
            print(fname + " file is not empty")
            embeddings = pd.read_csv(fname, header=None).as_matrix() # Get embeddings as a matrix from reps.csv
        else:
            print(fname + " file is empty")
            embeddings = np.zeros((2,150)) #creating an empty array since csv is empty

        self.le = LabelEncoder().fit(labels) # LabelEncoder is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1
        # Fits labels to model
        labelsNum = self.le.transform(labels)
        nClasses = len(self.le.classes_)
        print("Training for {} classes.".format(nClasses))

        if classifier == 'LinearSvm':
            self.clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier == 'GMM':
            self.clf = GMM(n_components=nClasses)

        if ldaDim > 0:
            clf_final =  self.clf
            self.clf = Pipeline([('lda', LDA(n_components=ldaDim))
                ('clf', clf_final)])

        self.clf.fit(embeddings, labelsNum) #link embeddings to labels

        fName = "{}classifier.pkl".format(workDir)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((self.le,  self.clf), f) # Creates character stream and writes to file to use for recognition
    #
    # def getSquaredl2Distance(self,rep1,rep2):
    #     """Returns number between 0-4, Openface calculated the mean between
    #     similar faces is 0.99 i.e. returns less than 0.99 if reps both belong
    #     to the same person"""
    #
    #     d = rep1 - rep2



if __name__ == "__main__":

    a = FaceRecogniser()
    # a.trainClassifier()
    aligndlib.alignMain("input/", "output/", "innerEyesAndBottomLip", "/home/hatice/PycharmProjects/newOpenface/models/dlib/shape_predictor_68_face_landmarks.dat", args.imgDim)

    a.recognize_face("output/bugraabi.png")

