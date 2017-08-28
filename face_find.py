import scipy
from PIL import Image
import face_recognition


def load_image_file(filename, mode='RGB'):
    return scipy.misc.imread(filename, mode=mode)

def face(img=""):
    image = face_recognition.load_image_file(img)

    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        top, right, bottom, left = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        face_image = image[top:bottom, left:right]
        # print(face_image)
        pil_image = Image.fromarray(face_image)
        # pil_image.show()
        # pil_image.save("yeni2.jpg")



