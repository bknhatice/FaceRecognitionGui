import pickle
import base64
import os
from PIL import Image
import imagehash

def deserialize(a):
    b = base64.b64decode(a)
    x = pickle.loads(b)
    return x

def serialize(a):
    x = pickle.dumps(a)
    b = unicode(base64.b64encode(x))
    return b

def images(dizin=""):
    image =[]
    for d in os.listdir(dizin):
        image.append(d)
    print image
    for i in image:
        path = "/home/hatice/PycharmProjects/newOpenface/input/" + i
        f = open(path, "rb").read()
        a = i.split(".")
        print a[0]
        b = a[0] + ".txt"
        text_file = open(b, "a")
        print("dosya acildi.")
        qq = base64.b64encode(f)
        yaz = a[0] + " isimli resim:(base64) " + qq
        text_file.write(yaz)
        text_file.close()
        # qq1 = base64.b64decode(qq)
        # f = open("/home/hatice/PycharmProjects/newOpenface/cem__1.jpg", "wb")
        # f.write(qq1)
        # f.flush()
        # f.close()
        # path = "/home/hatice/PycharmProjects/newOpenface/input/" + i
        # hash = imagehash.dhash(Image.open(path))
        # print(i + "hash i hesaplandi.")
        # img = serialize(hash)
        # print(i + " donusturuldu")
        # yaz = i + " isimli resim(base64): " + img + "\n"
        # text_file.write(yaz)
        # print(i + "dosyaya yazildi.")
        # text_file.close()
def image(hash=""):

    d = deserialize(hash)
    print d

if __name__ == "__main__":
    # images("/home/hatice/PycharmProjects/newOpenface/input")

    f = open("/home/hatice/PycharmProjects/newOpenface/input/gulen.jpg", "rb").read()

    qq = base64.b64encode(f)
    qq1 = base64.b64decode(qq)
    f = open("/home/hatice/PycharmProjects/newOpenface/cem__1.jpg", "wb")
    f.write(qq1)
    f.flush()
    f.close()



