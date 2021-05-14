import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('face.xml')
i = 0
offset = 50
kisi_id = input("ID bilgisi giriniz: ")
while True:
    ret,image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5,minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        i+=1
        cv2.imwrite("yuzverileri/face-" + kisi_id + "." + str(i) + ".jpg",gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(image,(x-offset,y-offset), (x+w+offset,y+h+offset),(0,255,255),3)
        cv2.waitKey(100)

    if i>20:
        cam.release()
        cv2.destroyAllWindows()
        break


