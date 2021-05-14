import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml')
cascadePath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = "yuzverileri"
cam  = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,minNeighbors=5)
    for (x,y,w,h) in faces:
        tahmin_edilen_kisi, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(image,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
        if (tahmin_edilen_kisi==1):
            tahmin_edilen_kisi = "Gül Cincik"
        elif (tahmin_edilen_kisi == 2):
            tahmin_edilen_kisi = "Aziz Sancar"
        else:
            tahmin_edilen_kisi = "NE BİLİM AMK"

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,0,255)
        cv2.putText(image,str(tahmin_edilen_kisi), (x,y+h), fontFace,fontScale, fontColor)
        cv2.imshow("resim", image)
        cv2.waitKey(10)