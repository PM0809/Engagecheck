import numpy as np
import os
import pandas as pd
import cv2

def image_seq_sort(pathes):
    pathes = np.array(pathes)
    idx_fn = lambda p : int(os.path.basename(p).split('.')[0])
    idx_fn_v = np.vectorize(idx_fn)
    idx = idx_fn_v(pathes)
    idx_sorted = np.argsort(idx)
    return pathes[idx_sorted]

class ImageChecker:
    def __init__(self,input_type):
        self.nw_state = None
        self.prev_img = None

        if input_type=='NW_image_seq':
            self.nw_state = pd.read_csv('NW_state.csv').state.values.tolist()

    def check(self,img):
        if self.nw_state is not None:
            state = self.nw_state.pop(0)
            return state
        else:
            state = self.is_blank_img(img) or self.is_repeated_img(img)
            return state
        
        
    def is_blank_img(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diffx = abs(gray[1:,:] - gray[:-1,:]).mean()
        diffy = abs(gray[:,1:] - gray[:,:-1]).mean()
        blank_ = diffx<5 or diffy<5
        return blank_
        
    def is_repeated_img(self,img):
        if self.prev_img is not None:
            repeated_ = (self.prev_img - img).sum()==0
            self.prev_img = img

            return repeated_
        else:
            self.prev_img = img
            return False

        