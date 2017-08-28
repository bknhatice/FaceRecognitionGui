import base64
import os, os.path
import random
import string
import pickle
from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
rsm = []
# you can use RFC-1738 to specify the url
app = Flask(__name__, static_folder="/home/hatice/PycharmProjects/newOpenface/static")

def img(nn = ""):
    es = Elasticsearch(['http://172.17.0.2:9200/'])
    response = es.search(
        index='new_imagesearch',
        body={
            "query": {
                "match": {
                    "_all": nn
                }
            }
        }
    )
    if nn != None :
        for result in response['hits']['hits']:
            a = result['_source']['image_binary']
            qq1 = base64.b64decode(a)
            p = "/home/hatice/PycharmProjects/newOpenface/static/" + nn + ".jpg"
            f = open(p, "wb")
            f.write(qq1)
            f.flush()
            f.close()
    elif nn == None:
        print "resim bulunamadi."



@app.route('/a')
def sayfa():
    return render_template('flask.html')

@app.route('/uploader',methods= ['POST','GET'])
def yukle():
    if request.method == 'POST' or request.method == 'GET':
        text = request.form['name']
        img(text)
        i = "./static/" + text + ".jpg"
        return render_template("flask.html", mesaj=text, image=i)


if __name__ == '__main__':
    app.run(debug=True, port=8090)