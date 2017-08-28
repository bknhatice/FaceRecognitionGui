import os
import random
import cv2
import face_recognition
# from caffe2.python.mint.app import args
from uuid import uuid4
import openface
from PIL import Image
from openface.data import iterImgs


def input_save(inputDir):

    imgs = os.listdir(inputDir)
    print imgs
    for img in imgs:
        path =  "input/" +str(img)

        numpy_image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(numpy_image)
        # print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for fl in face_locations:
            top, right, bottom, left = fl
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                      right))
            # You can access the actual face itself like this:
            face_image = numpy_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            #
            # pil_image.show()
            c = str(uuid4()).replace("-", ""[0:6])
            img_list = img.split(".")[0]

            if os.path.isdir(os.path.join("face",img_list)) == False:
                os.mkdir(os.path.join("face", img_list))

            yol = "face/" + img_list + "/" + c + ".jpg"
            pil_image.save(yol)

def alignMain(inputDir,output, landmarks, dlibFacePredictor,size):
    # input_save("input/")
    imgs = list(iterImgs(inputDir))
    name = inputDir.split("/")[1]
    output = os.path.join(output, name)
    # print name
    # print output
    # exit(-1)

    openface.helper.mkdirP(output)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    # innerEyesAndBottomLip daha duzgun aliyor.
    if landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmarkIndices = landmarkMap[landmarks]

    align = openface.AlignDlib(dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        path = "face/hepsi/" + str(imgObject) + ".png"

        # print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(output, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                # print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(size, rgb, landmarkIndices=landmarkIndices)
                if outRgb is None:
                    print("  + Unable to align.")

            if outRgb is not None:
                # print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)


if __name__ == "__main__":
    imgDim = 96
    # input_save("input/")
    # alignMain("face/hepsi/", "output/", "innerEyesAndBottomLip", "/home/hatice/PycharmProjects/newOpenface/models/dlib/shape_predictor_68_face_landmarks.dat", imgDim)

