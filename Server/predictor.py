import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image):
    Name=[]
    imgs = get_cropped_image_if_2_eyes(image)
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        Name=class_number_to_name(__model.predict(final)[0])
    return Name;
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cropped_image_if_2_eyes(image):
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

if __name__ == '__main__':
    name="none"
    load_saved_artifacts()
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            name = classify_image(frame)
    frame = cv2.imshow("Capturing", frame)
    cv2.waitKey(1)
    print(name)
    name="none"
    #print(classify_image("R.jpeg"))
    # print(classify_image(None, "./test_images/virat1.jpg"))
    # print(classify_image(None, "./test_images/virat2.jpg"))
    # print(classify_image(None, "./test_images/virat3.jpg"))
    # print(classify_image(None, "./test_images/serena1.jpg"))
    # print(classify_image(None, "./test_images/serena2.jpg"))
    # print(classify_image(None, "./test_images/sharapova1.jpg"))
