import base64
from datetime import datetime
import imagehash
import os
from PIL import Image
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://172.17.0.2:9200/'])
binarys = []
imagenames = []

def bin():
    dizin="/home/hatice/PycharmProjects/faceWeb/input/"
    image = []
    for d in os.listdir(dizin):
        image.append(d)
    for i in image:
        path = "/home/hatice/PycharmProjects/faceWeb/input/" + i
        f = open(path, "rb").read()
        iname = i.split(".")
        # print iname
        # exit(-1)
        qq = base64.b64encode(f)
        aa = "'" + qq + "'"
        binarys.append(aa)
        imagenames.append(iname)
        imagename = "'" + iname[0] + "'"
        print imagename
        con = input("resmin confidence : ")
        # print con
        # per = str(input("kim : "))
        # person = "'" + str(per) + "'"
        # print person
        # exit(-1)
        doc = {
            'image_name': imagename,
            'image_binary': aa,
            'resutls': {"confidence1": con, "person1": imagename},
        }
        res = es.index(index="new_imagesearch", doc_type='new_isr', body=doc)
        print(res['created'])
        es.indices.refresh(index="new_imagesearch")

# def kaydet():
#     f = open("/home/hatice/PycharmProjects/faceWeb/input/trump.jpg", "rb").read()
#     qq = base64.b64encode(f)
#     aa = "'" + qq + "'"
#     doc = {
#         'image_name': 'denemeTrump',
#         'image_binary': "",
#         'resutls': {"confidence1":98, "person1":"denemeTrump"},
#     }
#     res = es.index(index="new_imagesearch", doc_type='new_isr', body=doc)
#     print(res['created'])
#     es.indices.refresh(index="imagesearch")

def getir():
    res = es.get(index="new_imagesearch", doc_type='new_isr', id="aksener.jpg")
    img = res['_source']['image_binary']
    qq1 = base64.b64decode(img)
    f = open("/home/hatice/PycharmProjects/faceWeb/gelen.jpg", "wb")
    f.write(qq1)
    f.flush()
    f.close()
    es.indices.refresh(index="imagesearch")

if __name__ == "__main__":

   bin()
   #  getir()