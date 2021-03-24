import cv2
import dlib
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

TOLERANCE = 0.5


class Camera:
    video_frame_encodings = []
    face_names = []
    def __init__(self, camera_number, name_encoding):
        self.camera_number = camera_number
        self.name_encoding = name_encoding
        cap = cv2.VideoCapture(self.camera_number)

        while True:
            _, frame = cap.read()
            img = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

            faces = detector(img)

            for face in faces:
                self.x1 = face.left()  # left point
                self.y1 = face.top()  # top point
                self.x2 = face.right()  # right point
                self.y2 = face.bottom()  # bottom point

                landmarks = predictor(image=img, box=face)

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y

                    cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)

            shape_faces = [predictor(image=img, box=face) for face in faces]
            encod = [np.array(face_recognition_model.compute_face_descriptor(img, face_pose, 1)) for face_pose in shape_faces]
            if encod:
                self.video_frame_encodings.append(encod[0])
                if len(self.name_encoding) > 0:
                    self.match(self.name_encoding)
        
            cv2.imshow(winname="Face", mat=frame)

            # Exit when escape is pressed
            if cv2.waitKey(delay=1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def match(self, name_encoding):
        self.name_encoding = name_encoding
        self.video_frame_encoding_temp = self.video_frame_encodings
        self.names = list(self.name_encoding.keys())
        self.encodings = list(self.name_encoding.values())

        for encoding in self.video_frame_encoding_temp:
            self.result = (np.linalg.norm(self.encodings - encoding, axis=1) <= TOLERANCE)
            self.face_distance =  np.linalg.norm(self.encodings - encoding, axis=1)
            self.best_match_index = np.argmin(self.face_distance)

            if self.result[self.best_match_index]:
                name = self.names[self.best_match_index]
                if name not in self.face_names:
                    self.face_names.append(name)
                    if len(self.name_encoding) > 0:
                        self.name_encoding.pop(name)

                        # print(self.getResult())
        print(self.face_names)


    def getResult(self):
        return self.face_names

name_encodings = {
    "Manish":[-0.07485977,  0.10935297,  0.0211652 , -0.01359143, -0.04174919,
       -0.07415639, -0.05137352, -0.10307552,  0.13580616, -0.05250587,
        0.20499082, -0.06007884, -0.2243752 , -0.06362929, -0.02235227,
        0.07903469, -0.16269591, -0.04537497, -0.10501571, -0.0374122 ,
        0.08376001,  0.05812337,  0.11240809,  0.02710691, -0.13660489,
       -0.37591848, -0.09162537, -0.18607895,  0.01611444, -0.07582068,
       -0.07390258, -0.0474506 , -0.18324994, -0.07777709, -0.00385102,
       -0.03295413, -0.08571519, -0.04172176,  0.2297883 ,  0.01263885,
       -0.14961824, -0.03267143,  0.01365876,  0.2381068 ,  0.15908766,
        0.06115829,  0.01165238, -0.11340644,  0.0706054 , -0.17372754,
        0.08651777,  0.15314907,  0.07803119,  0.05821056,  0.08328515,
       -0.08546992,  0.02859398,  0.0489618 , -0.21621627,  0.03633688,
        0.07682868, -0.02676922, -0.0511745 , -0.03640967,  0.2247467 ,
        0.07543686, -0.09274448, -0.07325581,  0.0730406 , -0.15398139,
        0.00297623,  0.04741799, -0.1172047 , -0.23401117, -0.25161457,
        0.01755182,  0.36025411,  0.15007481, -0.14964418,  0.00186406,
       -0.10710815, -0.10240463,  0.10941382,  0.04419379, -0.08772468,
        0.05876616, -0.07321789,  0.13437559,  0.14519039, -0.10754672,
       -0.02812318,  0.22300519, -0.02090607,  0.04258406,  0.02072548,
       -0.00566322, -0.13832289,  0.01506463, -0.10497658,  0.02543873,
        0.1094256 , -0.03307442,  0.01449874,  0.06351475, -0.13225169,
        0.0826672 ,  0.02711518, -0.09680051, -0.00861181, -0.06529045,
       -0.15961817, -0.09793691,  0.14258513, -0.17695463,  0.19553642,
        0.12862103,  0.05722181,  0.13832203,  0.03717271,  0.00381259,
       -0.02825347, -0.00373556, -0.18948607, -0.00697943,  0.12196741,
        0.03568948,  0.11048013, -0.02830423],
        "Trump":[-0.10337359,  0.11473114,  0.07074415, -0.03220323, -0.13627155,
       -0.01554937,  0.01243954, -0.18070652,  0.07694289, -0.11361603,
        0.23251016, -0.07616171, -0.287135  , -0.07403055, -0.03633641,
        0.14055654, -0.11898371, -0.09958634, -0.15481851, -0.15858422,
        0.02405614,  0.04012005,  0.02020358, -0.03504929, -0.10914959,
       -0.22270967, -0.06325727, -0.04779207,  0.01201388, -0.1171167 ,
        0.07703294,  0.061433  , -0.23231149, -0.05356783, -0.0298068 ,
        0.10141789, -0.07702114, -0.08190008,  0.10985917, -0.03071729,
       -0.09231644, -0.01163826, -0.02643724,  0.19609365,  0.20151001,
       -0.03129319, -0.01217721, -0.1303996 ,  0.10631116, -0.22322717,
       -0.01538152,  0.09875514,  0.05113885,  0.10029927,  0.01747455,
       -0.06512391,  0.07507667,  0.14861006, -0.1383296 ,  0.02221187,
        0.05935117, -0.25578892, -0.05510892, -0.05109918,  0.08317305,
        0.06464259,  0.01589913, -0.14623818,  0.22781892, -0.13488966,
       -0.14515287,  0.03665665, -0.05233146, -0.10093244, -0.32650295,
       -0.00886125,  0.38184598,  0.11442128, -0.19316325, -0.01624732,
       -0.05620838,  0.01648931,  0.07032919,  0.03430058, -0.02543113,
       -0.11338006, -0.14754222,  0.0620675 ,  0.23272482, -0.11574952,
       -0.04863069,  0.21232685,  0.01183429, -0.00089102,  0.07905817,
        0.02104289, -0.07890537,  0.02062244, -0.07487009, -0.04234015,
        0.09578493, -0.14986557, -0.02235897,  0.08635416, -0.14022475,
        0.21425992, -0.01740149, -0.00133799, -0.00060789, -0.12307512,
       -0.01959008,  0.06550429,  0.19946326, -0.18794614,  0.24951814,
        0.21373138, -0.03688174,  0.1558    ,  0.04200245,  0.10875222,
       -0.05431583, -0.05400551, -0.16776079, -0.1523784 ,  0.01935603,
        0.03824571, -0.00069619,  0.06378622],
        "PM Narandra Modi":[-0.1521814 ,  0.06035386,  0.072544  , -0.12395531, -0.01336331,
       -0.13173665,  0.06602281, -0.0643186 ,  0.17531365,  0.01599292,
        0.24119832, -0.0333885 , -0.23747422, -0.10424302,  0.06714723,
        0.05789172, -0.15345268, -0.15199894, -0.10715617, -0.14065778,
        0.01255426, -0.05043061,  0.01280546,  0.07882877, -0.07767326,
       -0.36747348, -0.13032201, -0.21092826,  0.05770615, -0.09330989,
        0.02453368,  0.01219036, -0.202989  , -0.09561753, -0.02724987,
        0.08084363, -0.02161556, -0.05195744,  0.15325032, -0.00836928,
       -0.10665226,  0.02947597,  0.02653228,  0.1925938 ,  0.22352071,
        0.10925633,  0.0109619 , -0.01624399,  0.05776639, -0.2519843 ,
        0.10754429,  0.15047976,  0.04526972,  0.07040734,  0.0819466 ,
       -0.15499328,  0.0253068 ,  0.02539204, -0.25042978,  0.05728134,
        0.06152085, -0.18715475, -0.12565146, -0.04423104,  0.17952111,
        0.16385458, -0.09588806, -0.0946802 ,  0.11289448, -0.19805469,
       -0.02735904,  0.053186  , -0.12270569, -0.07822835, -0.24571998,
        0.1627831 ,  0.4365446 ,  0.11599912, -0.11525891,  0.03225321,
       -0.17860508,  0.00241282,  0.00370814, -0.03477914, -0.11143715,
       -0.08155365, -0.17369694,  0.02426145,  0.07933992,  0.00556535,
        0.00952603,  0.13634551, -0.10336098,  0.09716488,  0.01774276,
        0.06937637, -0.14880171, -0.02894349, -0.02990784, -0.04445093,
        0.09750868, -0.09655543, -0.03158798,  0.03090671, -0.17457104,
        0.07084784, -0.02799773, -0.06751986, -0.02769412, -0.00078543,
       -0.00953512, -0.0428353 ,  0.11806595, -0.23345119,  0.25629199,
        0.22061515, -0.03823647,  0.16249928,  0.01952   ,  0.07022204,
       -0.08740736, -0.04052413, -0.1487098 , -0.05373386,  0.01136572,
        0.09624682,  0.03829999,  0.02640571],
        "Vijay":[-0.17015727,  0.10171774,  0.02392299, -0.07260167, -0.18501034,
       -0.05237048, -0.00891863, -0.10110465,  0.16759416, -0.01166694,
        0.25239027, -0.02456139, -0.14217749, -0.07167197, -0.02916877,
        0.15774074, -0.09140366, -0.13458121, -0.10535977, -0.06852258,
        0.02728308,  0.00887077,  0.00881175,  0.1016417 , -0.18242   ,
       -0.31545779, -0.08140579, -0.13195083, -0.01917214, -0.06934574,
        0.01602831, -0.01907103, -0.17492077, -0.11124641, -0.00549597,
        0.03868048, -0.0177495 , -0.04086337,  0.18218228, -0.01601121,
       -0.16464445,  0.06088557,  0.04953605,  0.26252127,  0.18351141,
        0.05786156,  0.09760386, -0.06899735,  0.07022193, -0.19294529,
        0.06534027,  0.10516982,  0.10598214, -0.010069  ,  0.04888992,
       -0.0792813 ,  0.01310106,  0.18909626, -0.22616319,  0.04504302,
        0.08246426, -0.01967852, -0.06425819, -0.08118898,  0.27047631,
        0.1546907 , -0.11182646, -0.08750702,  0.13953196, -0.13416158,
       -0.09833615, -0.0391503 , -0.12895602, -0.17885782, -0.32960805,
        0.07293103,  0.426025  ,  0.16668572, -0.26367825,  0.01854438,
       -0.09122457,  0.06217458,  0.05585836,  0.03209908, -0.05708626,
       -0.07250857, -0.14227623, -0.01394313,  0.15398268, -0.00365434,
        0.0269143 ,  0.20721196,  0.0645372 , -0.02977725,  0.01302181,
        0.06838895, -0.16003618,  0.015822  , -0.02496303,  0.01043942,
        0.04215246,  0.02375985,  0.06296676,  0.1090318 , -0.18044782,
        0.19321375, -0.0519192 , -0.003074  , -0.04868551,  0.04229331,
        0.00521382, -0.07659451,  0.11513101, -0.1690907 ,  0.14599331,
        0.13707764, -0.06448889,  0.18005805,  0.05821312,  0.09494729,
       -0.04705684,  0.07814887, -0.16187862, -0.05847569,  0.03225359,
       -0.09066482,  0.12792392,  0.10943027],
       "Vishnu":[-0.15235841,  0.10360015,  0.04389173, -0.11971863, -0.07220386,
        0.0266505 ,  0.00505329, -0.10173375,  0.19388954, -0.07810341,
        0.25890797, -0.01107267, -0.16921772, -0.13363454, -0.01534365,
        0.17203397, -0.13943382, -0.15111661, -0.00425695, -0.04858971,
        0.06821641, -0.02292328,  0.00355781,  0.11293479, -0.20683356,
       -0.35947806, -0.09840214, -0.1345354 ,  0.00121382, -0.1234675 ,
       -0.00528011,  0.07927395, -0.24428843, -0.03466585, -0.0286735 ,
        0.09345932, -0.01178182,  0.0046843 ,  0.13485654,  0.03099694,
       -0.18483165,  0.02583882,  0.04515232,  0.26532131,  0.17940517,
        0.02993793,  0.01449566, -0.05583317,  0.0233038 , -0.2200754 ,
        0.07426794,  0.08602156,  0.09604523,  0.06036313, -0.03568067,
       -0.1200214 ,  0.03046495,  0.07512603, -0.27549815,  0.04982547,
        0.04723421, -0.12453976, -0.09413187,  0.01496734,  0.2415771 ,
        0.14770639, -0.15527679, -0.06069187,  0.15128389, -0.11246253,
        0.01884269,  0.05495356, -0.10246153, -0.20432779, -0.2746588 ,
        0.09245618,  0.43906948,  0.1745072 , -0.22482906,  0.00921487,
       -0.14594944,  0.02787219,  0.05782378,  0.06843834, -0.06207856,
       -0.0115294 , -0.08029638,  0.10397556,  0.1177946 ,  0.02938178,
       -0.00555889,  0.26732057,  0.01902889,  0.05539248,  0.00518681,
        0.10709338, -0.08382019, -0.01486168, -0.09839718, -0.04710659,
       -0.00377591, -0.00644752,  0.00713104,  0.11646181, -0.21602543,
        0.22445248,  0.05677075, -0.04507828, -0.00835425,  0.03841294,
       -0.03503362, -0.06125201,  0.08937052, -0.2668511 ,  0.15639989,
        0.18960872,  0.02063826,  0.19452028,  0.08293813,  0.04799721,
       -0.01077894, -0.12178244, -0.16275229,  0.00355148,  0.11140574,
       -0.08051293,  0.12514921,  0.02475019]
}


c1 = Camera(0, name_encodings)

# c2 = Camera(0, name_encodings)
# c3 = Camera(0, name_encodings)
# c4 = Camera(0, name_encodings)








