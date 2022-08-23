import cv2
import imutils
import random
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path) # detect
emotion_classifier = load_model(emotion_model_path, compile=False) #save model
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)#get access to webcam
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300) #resizing image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert image from one color to other
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #scaleFactor – Parameter specifying how much the image size is reduced at each image scalE,
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    #minSize – Minimum possible object size. Objects smaller than that are ignored.
    canvas = np.zeros((250, 300, 3), dtype="uint8") #returns a new array of given shape and type, where the element's value as 0, #uint8 data type contains all whole numbers from 0 to 255
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
       # time.sleep(2)
    else: continue

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    print('Your recognized emotion is',label)
    cv2.imshow('Music player with Emotion recognition', frameClone)
   # time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

    mp = 'C:/Program Files (x86)/Windows Media Player/wmplayer.exe'
    if label=="happy":
        import happy
        randomfile = random.choice(os.listdir("D://VIT//emotionrecognitionpycharm//songs//happy"))
        print("You need happy mood songs, I'll play song for you:"+ randomfile)
        file = ('D://VIT//emotionrecognitionpycharm//songs//happy' + randomfile)
        subprocess.call([mp,file])
        continue
    elif label=="sad":
       import sad
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/sad"))
        print("You need sad mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/sad' + randomfile)
        subprocess.call([mp, file])
        continue
    elif label == "angry":
        import angry
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/angry"))
        print("You need angry mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/angry' + randomfile)
        subprocess.call([mp, file])
        continue
    elif label == "surprised":
        import surprise
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/surprise"))
        print("You need surprise mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/surprise' + randomfile)
        subprocess.call([mp, file])
        continue
    elif label == "disgust":
        import disgust
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/disgust"))
        print("You need disgust mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/disgust' + randomfile)
        subprocess.call([mp, file])
        continue
    elif label == "neutral":
        import neutral
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/neutral"))
        print("You need neutral mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/neutral' + randomfile)
        subprocess.call([mp, file])
        continue
    elif label == "scared":
        randomfile = random.choice(os.listdir("D:/VIT/emotionrecognitionpycharm/songs/scared"))
        print("You need scary mood songs, I'll play song for you:" + randomfile)
        file = ('D:/VIT/emotionrecognitionpycharm/songs/scared' + randomfile)
        subprocess.call([mp, file])
        continue
        

camera.release()
cv2.destroyAllWindows()


