# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:10:56 2018
Copyright © Mon Aug  6 09:10:56 2018 by Ranly
@author: Ranly
@E-mail：1193932296@qq.com
        I PROMISE
"""
import cv2
import numpy as np
import os
import sys

filename = 'eg.jpg'
pathface = 'F:/Ranly_Obj/openCV/haarcascade_frontalface_default.xml'
patheye = 'F:/Ranly_Obj/openCV/haarcascade_eye_tree_eyeglasses.xml'

def video_detect():
    
    face_cascade =cv2.CascadeClassifier(pathface)
    eye_cascade = cv2.CascadeClassifier(patheye)
    img = cv2.imread(filename)
    cap = cv2.VideoCapture(0)
    while(True):
        status,frame = cap.read()
        #print(frame.shape)
        if status:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img,(600,600),interpolation = cv2.INTER_CUBIC) 

            faces = face_cascade.detectMultiScale(gray, 1.3,5)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face_located = frame[y:y+h,x:x+w]
                gray_located = gray[y:y+h,x:x+w]
                eyes = eye_cascade.detectMultiScale(gray_located,1.03,5,0,(40,40))
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(face_located,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            frame = cv2.flip(frame,1)
            cv2.imshow('--Detected--',frame)
        else:
            img = cv2.resize(img,(480,640),interpolation = cv2.INTER_CUBIC)
            cv2.imshow("--Detected--",img)  
           # cv2.namedWindow('MrWang’s handsome face Detected!!')
        if cv2.waitKey(int(1000/24))&0xFFFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def generate():
    face_cascade = cv2.CascadeClassifier(pathface)
    camera = cv2.VideoCapture(0)
    count = 150
    while(1):
        status,frame = camera.read()
        if status:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3,5)
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                pic = cv2.resize(gray[y:y+h,x:x+w],(200,200),interpolation = cv2.INTER_CUBIC)
                count += 1
                cv2.imwrite('F:/Ranly_Obj/openCV/picture/%s.pgm'%str(count),pic)
            cv2.imshow('Generate picture........',frame)
        if cv2.waitKey(int(1000/12))&0xFFFF == ord(' '):
            break
    camera.release()
    cv2.destroyAllWindows()
    
def read_images(path  = 'F:/Ranly_Obj/openCV/picture'):  #path,size=(200,200)
    X = []
    y = []
    size=(200,200)
    for filename in os.listdir(path):
        try:
            pic_path = os.path.join(path,filename)
            
            if   int(filename[0]) == 1:
                y.append(0)
            elif int(filename[0]) == 2:
                y.append(1)
            elif int(filename[0]) == 3:
                y.append(2)    
            elif int(filename[0]) == 4:
                y.append(3)  
            else:
                pass
            
            face = cv2.imread(pic_path,0)
            face = cv2.resize(face,size)
            X.append(np.asarray(face,dtype = np.uint8))  #gray=uint8  
            
       # except IOError, (errno,strerror):
       #     print('I/O error({0}):{1}'.format(errno,strerror))
        except:         
            print('Unexpected Error:',sys.exc_info()[0])
            raise

    return [X,y]
            
def face_recgonize():
    name = ['xujianmiao','wangchao','wangxu','yanzheng']          
    [X,y] = read_images()
    y = np.asarray(y,dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(X),np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(pathface)
    while(1):
        status,frame = camera.read()
        frame = cv2.flip(frame,1)
        #if (status):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3,5)
        for (x,y,w,h) in faces:
            
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_gray = gray[y:y+h,x:x+w]
            try:
                face_gray = cv2.resize(face_gray,(200,200),interpolation = cv2.INTER_CUBIC)
                params = model.predict(face_gray)
                #frame = cv2.flip(frame,1)
                print('LABEL:%s,cofidence=%.2f'%(name[params[0]],params[1]))
                cv2.putText(frame,name[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
            except:
                pass#frame = cv2.flip(frame,1)
        
        cv2.imshow('face_recgonize',frame)
        if cv2.waitKey(int(1000/24))&0xFFFF == ord(' '):
            break              
    camera.release()
    cv2.destroyAllWindows()
                
                
if __name__  == '__main__':
     face_recgonize()
     #video_detect()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    