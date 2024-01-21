from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
img = "problem.webp"
img = cv2.imread(img)
model = YOLO("weights/best.pt")
results = model.predict(img)
for result in results:
    annotator = Annotator(img)
    boxes = result.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
        
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)   
# Wait for the user to press a key
cv2.waitKey(0)
 
# Close all windows
cv2.destroyAllWindows()