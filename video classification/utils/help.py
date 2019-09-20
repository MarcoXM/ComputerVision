from glob import glob


def getClass(PATH = '/kaggle/input/ucf101/ucf101/UCF-101/'):
    video_class = [i.split('/')[-2] for i in glob(PATH + '*/')]
    return vedio_class
    
def num_each_class(class_list,PATH):
    a = {}
    for i in class_list:
        a[i] = len(glob(PATH+i+'/*.avi'))
    return a
    
flatten = lambda l: [item for sublist in l for item in sublist]

def train_test_split(class_list, test_size = 0.2):
    train_list, test_list = [], []
    
    for i in class_list:
        data = glob(PATH+i+'/*.avi')
        length = len(glob(PATH+i+'/*.avi'))
        tridx = int(length*(1-test_size))
        train, test = data[:tridx],data[tridx:]
        train_list.append(train)
        test_list.append(test)
    
    return flatten(train_list), flatten(test_list)
    
if __name__ == '__main__':

    a,b = train_test_split(video_class, test_size = 0.2)
    print(len(a))
    
import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
%matplotlib inline
from tqdm import tqdm_notebook as tqdm

def getImages(trainFile,newPATH='train_1/'):
    if not os.path.exists(newPATH):
        os.makedirs(newPATH)

    for i in tqdm(range(len(trainFile))):
        count = 0
        videoFile = trainFile[i]
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                filename = newPATH + videoFile.split('/')[-2] +"_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()
    
    

    
    