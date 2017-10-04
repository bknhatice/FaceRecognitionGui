# FaceRecognitionGui
In this project:

*Find who you are in given pictures. 
*Picture training is also done.
*In addition, ElasticSearch is recorded and displayed

**To find out who in the picture:
*/input folder is given an image.
face_align.recognizer.py in the main part are written below:
     a = FaceRecogniser()
     new_aligndlib.input_save("input/")
     new_aligndlib.alignMain("face/", "output/", "innerEyesAndBottomLip",
                        "/home/hatice/PycharmProjects/newOpenface/models/dlib/shape_predictor_68_face_landmarks.dat",
                        args.imgDim)
python face_align_recognizer.py
 
**Picture training:
*/img folder is given are images.
face_align.recognizer.py in the main part are written below:
     a = FaceRecogniser()
     a.trainClassifier()
python face_align_recognizer.py

** For elasticsearch
python elsq.py 

Work on the project continues.. They have shortcomings..
