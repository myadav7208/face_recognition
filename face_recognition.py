from __future__ import print_function
import recognizer as rc
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import itertools
from flask_cors import CORS, cross_origin
import json
from json import JSONEncoder
import cv2
from urllib.request import Request, urlopen
from timeit import default_timer as timer




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
    except Exception as e:
        return jsonify({"status":503, "description":str(e)}), 503






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
    except Exception as e:
        return jsonify({"status":503, "description":str(e)}), 503



class multiFace:
    def __init__(self, camera_number, known_face_names, known_face_encodings):
        self.camera_number = camera_number
        self.known_face_encodings = known_face_encodings

        self.known_face_names = known_face_names
        self.known_face_encodings = known_face_encodings
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def startRecognition(self):
        start = timer()
        face_names = []
        self.video_capture = cv2.VideoCapture(self.camera_number)
        if self.video_capture is None or not self.video_capture.isOpened():
            return str(self.camera_number)+" Camera Not Working"
        while True:
            # Grab a single frame of video
            ret, frame = self.video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_names_temp = []
            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = rc.face_locations(rgb_small_frame)
                face_encodings = rc.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = rc.compare_faces(self.known_face_encodings, face_encoding)
                    name = "unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                        if name not in face_names:
                            face_names.append(name)

                    # Or instead, use the known face with the smallest distance to the new face
                    # face_distances = rc.face_distance(self.known_face_encodings, face_encoding)
                    # best_match_index = np.argmin(face_distances)
                    # if matches[best_match_index]:
                    #     print("matches")
                    #     name = self.known_face_names[best_match_index]
                    face_names_temp.append(name)
                        

            self.process_this_frame = not self.process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names_temp):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if timer() - start >= 120 or cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()
        return face_names


@app.route('/multiple-match', methods=["POST"])
def classroomRecognition():
    if request.method == 'POST':
        token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MSwibmFtZSI6Inh5eiIsInBlcm1pc3Npb25fc3RyaW5nIjoiY3J1ZCIsImlhdCI6MTYxNzc5NjA3N30.raoZ1nupE-ak_zYp28b3msRetXjNDbwKeGepYR1JWLc'
        if request.form.get('dept_id'):
            known_face_names = []
            known_face_encodings = []
            #  and request.form.get('acad_year') and request.form.get('sem') and request.form.get('divs')
            req = Request('https://api-attendance-test.herokuapp.com/api/students')
            req.add_header('x-auth-token', token)
            content = json.loads(urlopen(req).read().decode())
            for i in range(len(content)):
                known_face_names.append(content[i]['roll_number'])
                decodedArrays = json.loads(content[i]['encoding'])
                finalEncoding = np.asarray(decodedArrays['encoding'])
                known_face_encodings.append(finalEncoding)

    
            multi = multiFace(1, known_face_names, known_face_encodings)
            presentRollNumbers = multi.startRecognition()
            print(presentRollNumbers)
            return jsonify({"status":"success", "description":str(presentRollNumbers)}), 200
        else:
            return jsonify({"status":"failed", "description":"bad"}), 404
        
    

    

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