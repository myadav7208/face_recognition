from flask import Flask, request, jsonify
import json
import os
from PIL import Image
from flask_cors import CORS, cross_origin
import recognizer
import numpy as np


app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'



# Function to get encoding
@app.route('/getEncoding', methods=['POST'])
@cross_origin()
def img_encoding():
    if request.method == 'POST':
        isthisFile = request.files.get('file')
        if isthisFile is not None:
            image = Image.open(isthisFile)
            if os.path.exists("tempImage/"):
                image.save("tempImage/"+isthisFile.filename+'.jpg')
            else:
                os.makedirs("tempImage")
                image.save("tempImage/"+isthisFile.filename+'.jpg')
            try:
                res = recognizer.face_register("tempImage/"+isthisFile.filename+'.jpg')
                if os.path.exists("tempImage/"+isthisFile.filename+'.jpg'):
                    os.remove("tempImage/"+isthisFile.filename+'.jpg')
                    if type(res) == int:
                        return jsonify({"status":"failed", "message":str(res)+" face Detected."})
                    else:
                        return jsonify({"status":"success", "encoding":str(res)})
            except:
                return jsonify({"status":"failed", "message":"Internal Error"})
        else:
            return jsonify({"status":"failed", "message":"Image file not found."})



# Function to match single image.
@app.route('/single_match', methods=["POST"])
@cross_origin()
def face_match():
    if request.method == 'POST':
        isthisFile = request.files.get('file')
        if isthisFile is not None:
            try:
                enc = request.form.get('enc')[1:-1:].replace("\n", '').split()
                enc = np.asarray(enc, dtype=np.float64) 
                image = Image.open(isthisFile)
                if os.path.exists("tempImage/"):
                    image.save("tempImage/"+isthisFile.filename+'.jpg')
                else:
                    os.makedirs("tempImage")
                    image.save("tempImage/"+isthisFile.filename+'.jpg')
                current_encoding = recognizer.face_register("tempImage/"+isthisFile.filename+'.jpg')
                if os.path.exists("tempImage/"+isthisFile.filename+'.jpg'):
                    os.remove("tempImage/"+isthisFile.filename+'.jpg')
                if type(current_encoding) == int:
                    return jsonify({"status":"failed", "message":str(current_encoding)+" face Detected."})
                else:
                    res = recognizer.compare_face_encodings([current_encoding], enc)
                    return jsonify({"status":"success", "message":str(res[0])})
            except:
                jsonify({"status":"failed", "message":"Internal Error."})
        else:
            jsonify({"status":"failed", "message":"Image file not find."})

    
# app.run(host='0.0.0.0', port='5001', debug=True)