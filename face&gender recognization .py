
import cv2
import numpy as np


FACE_PROTO = "weights/deploy.prototxt.txt"

FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

GENDER_MODEL = 'weights/deploy_gender.prototxt'
 
GENDER_PROTO = 'weights/gender_net.caffemodel'
 
MODEL_MEAN_VALUES = (78.426337760, 87.7689143744, 114.895847746)

GENDER_LIST = ['Male', 'Female']


AGE_MODEL = 'weights/deploy_age.prototxt'

AGE_PROTO = 'weights/age_net.caffemodel'

AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

frame_width = 1280
frame_height = 720

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
   
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
   
    face_net.setInput(blob)
    
    output = np.squeeze(face_net.forward())
    
    detectedfaces = []
   
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
        
            start_x, start_y, end_x, end_y = box.astype(np.int)
         
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
     
            faces.append((start_x, start_y, end_x, end_y))
    return faces


def display_img(title, img):
    """Displays an image on screen and maintains the output until the user presses a key"""
   
    cv2.imshow(title, img)
    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


 
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
 
    dim = None
    (h, w) = image.shape[:2]
   
    if width is None and height is None:
        return image
  
    if width is None:
        
        r = height / float(h)
        dim = (int(w * r), height)
    
    else:
     
        r = width / float(w)
        dim = (width, int(h * r))
 
    return cv2.resize(image, dim, interpolation = inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()



def predict_age_and_gender(input_path: str):
    """Predict the gender of the faces showing in the image"""
   
    img = cv2.imread(input_path)
 
    frame = img.copy()
    if frame.shape[1] > frame_width:
 
    faces = get_faces(frame)
  
    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        age_preds = get_age_predictions(face_img)
        gender_preds = get_gender_predictions(face_img)
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        age_confidence_score = age_preds[0][i]
  
        label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
        
        print(label)
        yPos = start_y - 15
        while yPos < 15:
            yPos += 15
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
     
        font_scale = 0.54
        cv2.putText(frame, label, (start_x, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 3)

     
    display_img("Gender Estimator", frame)"loop"
  
    cv2.imwrite("output.jpg", frame)
   
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    predict_age_and_gender(input_path)
   
import cv2
import numpy as np
 
GENDER_MODEL = 'weights/deploy_gender.prototxt'
 
GENDER_PROTO = 'weights/gender_net.caffemodel'
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
 
GENDER_LIST = ['Male', 'Female' , 'Transgender']
 
FACE_PROTO = "weights/deploy.prototxt.txt"
 
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
 
AGE_MODEL = 'weights/deploy_age.prototxt'
 
 
AGE_PROTO = 'weights/age_net.caffemodel'
 
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

frame_width = 1280
frame_height = 720

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    
    face_net.setInput(blob)
    
    output = np.squeeze(face_net.forward())
    
    faces = []
    
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                np.assertg
                         frame.shape[1], frame.shape[0]])
        
            start_x, start_y, end_x, end_y = box.astype(np.int)

            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            
            faces.append((start_x, start_y, end_x, end_y))
    return faces



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    
    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        
        r = height / float(h)
        dim = (int(w * r), height)
    
    else:
        
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation = inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()



def predict_age_and_gender():
    """Predict the gender of the faces showing in the image"""
    
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        
        frame = img.copy()
        
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        
        faces = get_faces(frame)
        
        
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
        
            age_preds = get_age_predictions(face_img)
            
            gender_preds = get_gender_predictions(face_img)

            i = gender_preds[0].argmax()

            gender = GENDER_LIST[i] 

            J = gender_preds[j]

            gender = GENDER_LIST[J]


            gender_confidence_score = gender_preds[0][i]
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            age_confidence_score = age_preds[0][i]
            
            label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
            lable = t"{gender}-{gender_confidence_scorce*100:.1t}%, {age}-{age_confidence_score*100:.1t}%"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, box_color, 2)

           
        cv2.imshow("Gender Estimator", frame)
        if cv2.waitKey(1) == ord("q"):
            break
      
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_age_and_gender()