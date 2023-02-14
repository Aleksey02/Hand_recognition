import cv2 
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import export_text

#cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin,volMax = volume.GetVolumeRange()[:2]
i=0
firstFinger=[5,6,7,8]

resFirst=0
resSecond=0
resThird=0
resFours=0
resFifth=0
secondFinger=[9,10,11,12]

thirdFinger=[13,14,15,16]

fourthFinger=[17,18,19,20]
fifthFinger=[4,3,2]
i=1
FingerResTrain = []
CurrentFingerResTrain = []
gest = ['call','like']#, 'like', 'ok', 'hello', 'peace','rock'
numberGest = 0
classif = []
numberClassif=[]
try:
    while True:
        if i>100:
            i=1
            numberGest+=1

        #success,img = cap.read()
        img=cv2.imread(f'./dataset/image/{gest[numberGest]}/photo ({i}).jpg')

        i += 1
        img = cv2.flip(img, 1)
        img=cv2.resize(img, (400,400))
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        countFinger = 0
        lmList = []
        fingers = []
        currentFinger = None

        resFinger = []
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                firstFingersum = []
                secondFingersum = []
                thirdFingersum = []
                fourthFingersum = []
                fifthFingersum = []
                fifthFingerDopsum=[]
                midLine=[]
                firstFingerOn=False
                secondFingerOn=False
                thirdFingerOn=False
                fourthFingerOn=False
                fifthFingerOn=False
                firstFingerCoord=None
                secondFingerCoord=None
                thirdFingerCoord=None
                fourthFingerCoord=None
                fifthFingerCoord=None

                h, w, _ = img.shape
                for id,lm in enumerate(handlandmark.landmark):


                    #print(id, lm)
                    if id==5 or id==8:
                        firstFingersum.append([lm.x*w, lm.y*h])
                    if id==9 or id==12:
                        secondFingersum.append([lm.x*w, lm.y*h])
                    if id==13 or id==16:
                        thirdFingersum.append([lm.x*w, lm.y*h])
                    if id==17 or id==20:
                        fourthFingersum.append([lm.x*w, lm.y*h])
                    if id==1 or id==3:#нужно исправить:
                        fifthFingersum.append([lm.x * w, lm.y * h])
                    if id==5 or id==17:
                        midLine.append([lm.x * w, lm.y * h])
                    if id==5 or id==3:
                        fifthFingerDopsum.append([lm.x * w, lm.y * h])
                    if id<=1:
                        fifthFingerDopsum.append([lm.x * w, lm.y * h])
                    if id==4:
                        fifthFingerCoord=[lm.x * w, lm.y * h]
                    if id==8:
                        firstFingerCoord=[lm.x * w, lm.y * h]
                    if id==20:
                        resFirst=(((firstFingersum[1][0]-firstFingersum[0][0])**2)+((firstFingersum[1][1]-firstFingersum[0][1]))**2)**0.5
                        resSecond=(((secondFingersum[1][0]-secondFingersum[0][0])**2)+((secondFingersum[1][1]-secondFingersum[0][1]))**2)**0.5
                        resThird=(((thirdFingersum[1][0]-thirdFingersum[0][0])**2)+((thirdFingersum[1][1]-thirdFingersum[0][1]))**2)**0.5
                        resFours=(((fourthFingersum[1][0]-fourthFingersum[0][0])**2)+((fourthFingersum[1][1]-fourthFingersum[0][1]))**2)**0.5
                        resFifth=(((fifthFingersum[1][0]-fifthFingersum[0][0])**2)+((fifthFingersum[1][1]-fifthFingersum[0][1]))**2)**0.5
                        resFifthDop1=(((fifthFingerDopsum[1][0]-fifthFingerDopsum[0][0])**2)+((fifthFingerDopsum[1][1]-fifthFingerDopsum[0][1]))**2)**0.5
                        resFifthDop2=(((fifthFingerDopsum[3][0]-fifthFingerDopsum[2][0])**2)+((fifthFingerDopsum[3][1]-fifthFingerDopsum[2][1]))**2)**0.5
                        resMid=(((midLine[1][0]-midLine[0][0])**2)+((midLine[1][1]-midLine[0][1]))**2)**0.5

                        if(resFirst>resMid):
                            countFinger+=1
                            fingers.append('1')
                            firstFingerOn=True
                        if(resSecond>resMid):
                            countFinger+=1
                            fingers.append('2')
                            secondFingerOn=True
                        if(resThird>resMid):
                            countFinger+=1
                            fingers.append('3')
                            thirdFingerOn=True
                        if(resFours+10>resMid):
                            countFinger+=1
                            fingers.append('4')
                            fourthFingerOn=True
                        if(resFifth+10>resMid and resFifthDop2>resFifthDop1):
                            countFinger+=1
                            fingers.append('5')
                            fifthFingerOn=True


                        if firstFingerOn and not (secondFingerOn or thirdFingerOn or fourthFingerOn or fifthFingerOn):
                            currentFinger='first finger'
                        if secondFingerOn and not (fifthFingerOn or thirdFingerOn or fourthFingerOn or firstFingerOn):
                            currentFinger='second finger'
                        if thirdFingerOn and not (secondFingerOn or fifthFingerOn or fourthFingerOn or firstFingerOn):
                            currentFinger='third finger'
                        if fourthFingerOn and not (secondFingerOn or thirdFingerOn or fifthFingerOn or firstFingerOn):
                            currentFinger='fourth finger'
                        if fifthFingerOn and not (secondFingerOn or thirdFingerOn or fourthFingerOn or firstFingerOn):
                            currentFinger='big finger'
                            if fifthFingersum[0][0]<fifthFingerCoord[0]<firstFingersum[0][0] or fifthFingersum[0][0]>fifthFingerCoord[0]>firstFingersum[0][0]:
                                if fifthFingerCoord[1]>fifthFingersum[0][1]:
                                    currentFinger='big finger down'
                                else:
                                    currentFinger='like'
                        if firstFingerOn and fourthFingerOn and not (secondFingerOn or thirdFingerOn or fifthFingerOn):
                            currentFinger='rock'
                        if firstFingerOn and secondFingerOn and not (thirdFingerOn or fourthFingerOn or fifthFingerOn):
                            currentFinger='peace'
                        if firstFingerOn and secondFingerOn and thirdFingerOn and fourthFingerOn and fifthFingerOn:
                            currentFinger='hello'
                            if (((firstFingerCoord[0]-fifthFingerCoord[0])**2)+((firstFingerCoord[1]-fifthFingerCoord[1]))**2)**0.5<(((fifthFingerDopsum[0][0]-fifthFingerDopsum[1][0])**2)+((fifthFingerDopsum[0][1]-fifthFingerDopsum[1][1]))**2)**0.5:
                                currentFinger='ok'
                        if fifthFingerOn and fourthFingerOn and not (secondFingerOn or thirdFingerOn or firstFingerOn):
                            currentFinger='call'


                        if currentFinger!=None:
                            FingerResTrain.append(currentFinger)
                            CurrentFingerResTrain.append(gest[numberGest])
                        classif.append([resFirst-resMid, resSecond-resMid, resThird-resMid,resFours-resMid,resFifth-resMid])
                        numberClassif.append(numberGest)

                        #resFinger = [str(firstFingerOn), str(secondFingerOn), str(thirdFingerOn), str(fourthFingerOn), str(fifthFingerOn)]


                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if id==8 or id ==12:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

        if lmList != []:
            x1,y1 = lmList[4][1],lmList[4][2]
            x2,y2 = lmList[8][1],lmList[8][2]

            cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

            length = hypot(x2-x1,y2-y1)

            vol = np.interp(length,[15,220],[volMin,volMax])
            #print(vol,length)
            # volume.SetMasterVolumeLevel(vol, None)

            # Hand range 15 - 220
            # Volume range -63.5 - 0.0
        cv2.waitKey(1)
        if resFinger:
            cv2.putText(img, ''.join(resFinger), (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 0)

        cv2.putText(img, currentFinger, (10, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 250), 0)
        cv2.putText(img, str(countFinger), (10, 280), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 0)
        cv2.putText(img, ', '.join(fingers), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 0)
        cv2.imshow('Image',img)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break
except:
    classif = np.array(classif)
    #classif = classif.reshape(-1,1)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(classif, numberClassif)
    classPredicted = clf.predict(classif)
    # for x,y in zip(numberClassif,classPredicted):
    #     print(x,y)
    # print(classPredicted)
    # print(numberClassif)
    tree.plot_tree(clf)
    r = export_text(clf)
    #(r)
    #print(FingerResTrain)
    #print(CurrentFingerResTrain)
    print(confusion_matrix(CurrentFingerResTrain,FingerResTrain, labels=['call', 'like', 'ok', 'hello', 'peace','rock']))






