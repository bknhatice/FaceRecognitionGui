import scipy

import face_find


def cimage(img = ""):

    if vehicle_find(img) == True:
        print "vehicle"
    elif face(img) == True:
        print "face"


def face(img=""):
    face_find.face(img)
    return True

def vehicle_find(img=""):
    if face_find.face(img) == True:
        print "face"
        return False



if __name__ == "__main__":
    cimage("cem.jpg")