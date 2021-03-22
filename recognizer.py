import dlib
import scipy.misc
import numpy as np
import os

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
TOLERANCE = 0.5



# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    # Load image using scipy
    try:
        image = scipy.misc.imread(path_to_image)
        # Detect faces using the face detector
        detected_faces = face_detector(image, 1)
        print(detected_faces)
        shapes_faces = [shape_predictor(image, face) for face in detected_faces]
        print(shapes_faces)
        # For every face detected, compute the face encodings
        return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
    except:
        pass



# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    try:
        return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)
    except:
        pass




def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)
    # Return the name of the first match
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    # Return not found if no match found
    return 'Not Found'

# face_encoding = []
# face_encoding.append(get_face_encodings("D:\\xampp\\htdocs\\projects\\Attendance\\face_recognition\\images\\vishnu.jpeg")[0])
# enc2 = get_face_encodings("D:\\xampp\\htdocs\\projects\\Attendance\\face_recognition\\images\\m2.jpg")[0]

# print(face_encoding)
# print(enc2.shape)
# print(compare_face_encodings(face_encoding, enc2))
# print(enc1)
# print(enc2)

def face_register(path_to_image):
    try:
        face_encoding_in_image = get_face_encodings(path_to_image)
        if len(face_encoding_in_image) < 1:
            return len(face_encoding_in_image)
        elif len(face_encoding_in_image) > 1:
            return len(face_encoding_in_image)
        else:
            return face_encoding_in_image[0]
    except:
        pass

    