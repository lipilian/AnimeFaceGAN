# %% import package for face detection 
import cv2 
import sys
import os.path
from glob import glob
from joblib import Parallel, delayed
# %% define detection function 
cascade_file = "./lbpcascade_animeface.xml"
if not os.path.isfile(cascade_file):
    raise RuntimeError("%s: not found!" % cascade_file)
cascade_detector = cv2.CascadeClassifier(cascade_file)

# %% define the detection function
def detect(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade_detector.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (48, 48)
    )
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y + h,x:x + w]
        face = cv2.resize(face, (96, 96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("./faces/" + save_filename, face)

# %% run face detection to crop the image 
if not os.path.exists('faces'):
    os.makedirs('faces')
filelist = glob('imgs/*.jpg')
for filename in filelist:
    detect(filename)

# %%
