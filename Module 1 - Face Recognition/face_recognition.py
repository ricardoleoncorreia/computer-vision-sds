# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def get_gray_frame(color_frame):
    return cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

def get_feature_coordinates(gray_frame, cascade, scale, neighbors):
    return cascade.detectMultiScale(gray_frame, scale, neighbors)

def get_region_of_interest(frame, coordinates):
    (x, y, w, h) = coordinates
    return frame[y:y+h, x:x+w]

def draw_rectangles_in_frame(frame, coordinates, rgb_color, width):
    (x, y, w, h) = coordinates
    cv2.rectangle(frame, (x, y), (x+w, y+h), rgb_color, width)

# Defining a function that will do the detections
def detect_face_and_eyes(color_frame):
    gray_frame = get_gray_frame(color_frame)
    faces = get_feature_coordinates(color_frame, face_cascade, 1.3, 5)

    for face_coordinates in faces:
        draw_rectangles_in_frame(color_frame, face_coordinates, (255, 0, 0), 2)

        roi_gray = get_region_of_interest(gray_frame, face_coordinates)
        roi_color = get_region_of_interest(color_frame, face_coordinates)

        eyes = get_feature_coordinates(roi_gray, eye_cascade, 1.1, 3)

        for eyes_coordinates in eyes:
            draw_rectangles_in_frame(roi_color, eyes_coordinates, (0, 255, 0), 2)

    return color_frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, color_frame = video_capture.read()
    canvas = detect_face_and_eyes(color_frame)
    cv2.imshow('Video', canvas)
    
    user_press_q_key = cv2.waitKey(1) & 0xFF == ord('q')
    if user_press_q_key:
        break

video_capture.release()
cv2.destroyAllWindows()