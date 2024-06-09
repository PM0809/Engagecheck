import numpy as np
import pandas as pd
import cv2
import glob
import tqdm
import yaml
import os

from vision_module import EBG_module
from utils import ImageChecker,image_seq_sort


with open("config.yaml","r") as file_object:
    yaml_file = yaml.load(file_object,Loader=yaml.SafeLoader)
    input_type = yaml_file['input_type']
    happy_th = yaml_file['happy_threshold']
    attendance_th = yaml_file['attendance_threshold']

if __name__=='__main__':
    if input_type!='live_camera':
        file_path = input('Please enter the video sample / image folder name: ')
        
    entry_name = input('Please enter the record name (to be referenced in the results csv file): ')
    tot_engagement = []
    tot_emotions = []
    detector = EBG_module()
    image_checker = ImageChecker(input_type)
    
    if input_type=='video':
        video = cv2.VideoCapture(file_path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print('-----processing video-----')
        for i in tqdm.tqdm(range(0,length)): 
            _, frame = video.read() 
            null_frame_ = image_checker.check(frame)
            if null_frame_:
                continue
            instant_eng = detector.get_engagement(frame)
            tot_engagement.append(instant_eng)
            tot_emotions.append(detector.emotion_type)
            
    elif input_type=='live_camera':
        video = cv2.VideoCapture(0)
        print('-----processing live camera-----')
        print("-----press 'q' to end-----")

        while True:
            _, frame = video.read() 
            null_frame_ = image_checker.check(frame)
            if null_frame_:
                continue
            instant_eng = detector.get_engagement(frame)
            cv2.putText(frame, detector.emotion_type, (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 0), 2)
            tot_engagement.append(instant_eng)
            tot_emotions.append(detector.emotion_type)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    else:
        
        img_pathes = image_seq_sort(glob.glob(file_path + '/*'))
        print('-----processing image sequence-----')
        for img_path in tqdm.tqdm(img_pathes): 
            frame = cv2.imread(img_path) 
            null_frame_ = image_checker.check(frame)
            if null_frame_:
                continue
    
            instant_eng = detector.get_engagement(frame)
            tot_engagement.append(instant_eng)
            tot_emotions.append(detector.emotion_type)

    happy_ratio = detector.happy_count/len(tot_engagement)
    happy_effect = 1 if happy_ratio<happy_th else 1-(happy_ratio -happy_th) 
    
    percentage_eng = np.mean(tot_engagement)*100*happy_effect
    attendance = percentage_eng > attendance_th*100
    attendance_dict = {True:'present',False:'absent'}
    
    emotions,emotion_count = np.unique(tot_emotions,return_counts=True)
    common_emotion = emotions[np.argmax(emotion_count)]
    common_emotion_percentage = emotion_count.max()*100/emotion_count.sum()
    
    if not os.path.isfile('results.csv'):
        pd.DataFrame({'name':[],'Engagement':[],'Attendance':[],'Common emotion':[],'Common emotion percentage':[]}).to_csv('results.csv',index=False)
    old_results = pd.read_csv('results.csv',index_col=None)
    new_results = old_results.append({'name':entry_name,'Engagement':f"{percentage_eng:.2f}%",
                                      'Attendance':attendance_dict[attendance],'Common emotion':common_emotion,
                                      'Common emotion percentage':f"{common_emotion_percentage:.2f}%"}, ignore_index=True)
    new_results.to_csv('results.csv',index=False)
    print(f"total engagement percentage = {percentage_eng:.2f}%")
    print(f"Attendance : {attendance_dict[attendance]}")
    print(f"Common emotion: {common_emotion}")
    print(f"Common emotion percentage:{common_emotion_percentage:.2f}%")
    

    

