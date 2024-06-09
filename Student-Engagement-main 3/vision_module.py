import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model

    
class EBG_module:

    def __init__(self):
        self.emotion_model = load_model('./pretrained/emotion_model_resnet.107-0.66.hdf5')
        self.emotion_face_detector = cv2.CascadeClassifier('./pretrained/haarcascade_frontalface_default.xml')

        self.face_detector = dlib.get_frontal_face_detector()
        self.eye_detector = dlib.shape_predictor("./pretrained/shape_predictor_68_face_landmarks.dat")

        self.happy_count = 0
        self.emotion_type = 'Neutral'

    def midpoint(self, p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    def blinking_effect(self, frame, eye_points, facial_landmarks):
        ratios =[]
        for pts in eye_points:
            left_point = (facial_landmarks.part(pts[0]).x, facial_landmarks.part(pts[0]).y)
            right_point = (facial_landmarks.part(pts[3]).x, facial_landmarks.part(pts[3]).y)
            center_top = self.midpoint(facial_landmarks.part(pts[1]), facial_landmarks.part(pts[2]))
            center_bottom = self.midpoint(facial_landmarks.part(pts[5]), facial_landmarks.part(pts[4]))
            
            hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
            ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
            
            ratio = ver_line_lenght / hor_line_lenght
            ratios.append(ratio)
        blinking_distraction = np.mean(ratios)< 0.22
        return blinking_distraction


    def gaze_effect(self, frame, eye_points, facial_landmarks, gray):

        gaze_tot = []
        for pts in eye_points:
            eye_region = np.array([ (facial_landmarks.part(pts[0]).x,
                                     facial_landmarks.part(pts[0]).y),
                                    (facial_landmarks.part(pts[1]).x,
                                     facial_landmarks.part(pts[1]).y),
                                    (facial_landmarks.part(pts[2]).x,
                                     facial_landmarks.part(pts[2]).y),
                                    (facial_landmarks.part(pts[3]).x,
                                     facial_landmarks.part(pts[3]).y),
                                    (facial_landmarks.part(pts[4]).x,
                                     facial_landmarks.part(pts[4]).y),
                                    (facial_landmarks.part(pts[5]).x,
                                     facial_landmarks.part(pts[5]).y)], np.int32)
    
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [eye_region], True, 255, 2)
            cv2.fillPoly(mask, [eye_region], 255)
            eye = cv2.bitwise_and(gray, gray, mask=mask)
            
            min_x = np.min(eye_region[:, 0])
            max_x = np.max(eye_region[:, 0])
            min_y = np.min(eye_region[:, 1])
            max_y = np.max(eye_region[:, 1])
            gray_eye = eye[min_y: max_y, min_x: max_x]
            
            _, threshold_eye = cv2.threshold(gray_eye, np.median(gray_eye), 255, cv2.THRESH_BINARY)
        
            w = threshold_eye.shape[1]
            left_side_threshold = threshold_eye[:,:int(w/2)]
            left_side_white = cv2.countNonZero(left_side_threshold)
            right_side_threshold = threshold_eye[:,int(w/2):]
            right_side_white = cv2.countNonZero(right_side_threshold)
    
            gaze_ratio = (left_side_white+.1) / (right_side_white+.1)
            gaze_tot.append(gaze_ratio)
        gaze_distraction = (np.mean(gaze_tot)>3) | (np.mean(gaze_tot)<0.3)
        return gaze_distraction

    def emotion_effect(self, gray):
        emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                    4: 'Sad', 5: 'Surprised', 6: 'Neutral'}

        class_weights_wrt_neutral = np.array([0.5,1,1,0.5,0.5,1,0.5])

        
        faces_coor = self.emotion_face_detector.detectMultiScale(gray, 1.3, 5)
        emotion_distraction = False
        
        if len(faces_coor) > 0:
            x, y, w, h = faces_coor[0]
            gray_face = gray[y:y+h,x:x+h]
            gray_face = cv2.resize(gray_face, (64, 64))
            gray_face = gray_face.reshape([-1, 64, 64, 1])
            gray_face = gray_face*2/255-1
            
            pred_prob = self.emotion_model.predict(gray_face,verbose=0)[0]
            pred_prob = np.array(pred_prob)
            neutral_prob = pred_prob[6]
            pred = np.argmax(pred_prob)
            
            if (neutral_prob > class_weights_wrt_neutral*pred_prob).all():
                pred = 6
                
            self.emotion_type = emotion_dict[pred]
            
            if self.emotion_type == 'Happy':
                self.happy_count+=1
                
            elif self.emotion_type != 'Neutral':
                emotion_distraction = True
                
        return emotion_distraction

    def get_engagement(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        engagement = False
        
        if faces.__len__()>0:
            face = faces[0]
            landmarks = self.eye_detector(gray, face)
            eye_points = [[36, 37, 38, 39, 40, 41],
                          [42, 43, 44, 45, 46, 47]]
            
            blinking_distraction = self.blinking_effect(frame,eye_points, landmarks)
            gaze_distraction = self.gaze_effect(frame,eye_points, landmarks, gray)
            emotion_distraction = self.emotion_effect(gray)
            distraction = blinking_distraction | gaze_distraction | emotion_distraction
            engagement = ~distraction

        return engagement
