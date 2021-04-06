from __future__ import print_function
import recognizer as rc
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import itertools
from flask_cors import CORS, cross_origin
import json
from json import JSONEncoder



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.route('/get-encoding', methods=['POST'])
@cross_origin()
def extractEncoding():
    try:
        if request.method == "POST":
            if 'file' not in request.files:
                return jsonify({"status":"failed", "description":"Image file not found."}), 404
            file = request.files['file']
            if file.filename == "":
                return jsonify({"status":"failed", "desciption":"file name not found."}), 404
            if file and allowed_file(file.filename):
                rgb_image = rc.load_image_file(file)
                encoding = rc.face_encodings(rgb_image)
                if len(encoding) < 1:
                    return jsonify({"status":400, "description":"Face not detected."}), 400
                elif len(encoding) > 1:
                    return jsonify({"status":400, "description":"Multiple face detected."}), 400
                else:
                    encoded_data = {'encoding' : encoding[0]}
                    json_encoded = json.dumps(encoded_data, cls=NumpyArrayEncoder)
                    return jsonify({"status":"success", "description":json_encoded}), 200
            else:
                return jsonify({"status":"failed", "description":"Image file not supported. please use png, jpg, jpeg"}), 400
    except:
        return jsonify({"status":503, "description":"Internal server error."}), 503






@app.route('/single-match', methods=["POST"])
def single_match():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return jsonify({"status":"failed", "description":"Image file not found."}), 404
            if 'enc' not in request.form:
                return jsonify({"status":"failed", "description":"encoding not found."}), 404
        
            file = request.files['file']
            if file.filename == "":
                return jsonify({"status":"failed", "desciption":"file name not found."}), 404

            if file and allowed_file(file.filename):
                # new encoding
                rgb_image = rc.load_image_file(file)
                encoding = rc.face_encodings(rgb_image)
                if len(encoding) < 1:
                    return jsonify({"status":400, "description":"Face not detected."}), 400
                elif len(encoding) > 1:
                    return jsonify({"status":400, "description":"Multiple face detected."}), 400
                else:
                    # already stored endong in database
                    repl = request.form.get('enc').replace('\\', '')
                    decodedArrays = json.loads(repl)
                    finalEncoding = np.asarray(decodedArrays['encoding'])
                    res = rc.compare_faces(encoding, finalEncoding)
                    return jsonify({"status":"success", "description":str(res[0])}), 200
    except:
        return jsonify({"status":503, "description":"Internal Server Error"}), 503



    

# def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
#     if number_of_cpus == -1:
#         processes = None
#     else:
#         processes = number_of_cpus

#     # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
#     context = multiprocessing
#     if "forkserver" in multiprocessing.get_all_start_methods():
#         context = multiprocessing.get_context("forkserver")

#     pool = context.Pool(processes=processes)

#     function_parameters = zip(
#         images_to_check,
#         itertools.repeat(known_names),
#         itertools.repeat(known_face_encodings),
#         itertools.repeat(tolerance),
#         itertools.repeat(show_distance)
#     )

#     pool.starmap(test_image, function_parameters)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)