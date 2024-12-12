# computer_vision_projects

## Capturing Video

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
## Procedure to Detect a Specific Color and Draw a Bounding Box Using OpenCV
### Step 1: Import Required Libraries

```python
import cv2
import numpy as np

```
### Step 2: Set Up Video Capture

```python
cap = cv2.VideoCapture(0)
```
### Step 3: Define the HSV Range for the Target Color
Lower range for yellow
Upper range for yellow
```python
yellow_lower = np.array([20, 100, 100])  
yellow_upper = np.array([30, 225, 225])
```

### Step 4: Process Each Frame
```python
ret, frame = cap.read()
```

### Step 5: Convert the Frame to HSV
```python
hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

### Step 6: Create a Mask for the Target Color
```python
mask = cv2.inRange(hsvImage, yellow_lower, yellow_upper)
```
### Step 7: Apply the Mask
```python
result = cv2.bitwise_and(frame, frame, mask=mask)
```

### Step 8: Display the Outputs
```python
cv2.imshow('Original', frame)
cv2.imshow('Mask', mask)
cv2.imshow('Yellow Detection', result)
```
### Step 9: Draw Bounding Boxes
To draw bounding boxes, detect contours in the mask
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
Iterate through contours and filter small areas
```python
for contour in contours:
    if cv2.contourArea(contour) > 500:  
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
### Step 10: Handle Exit Condition
```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
### Step 11: Release Resources
```python
cap.release()
cv2.destroyAllWindows()
```


## Detecting Colors

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

yellow_lower = np.array([20,100,100])
yellow_upper = np.array([30,225,225])

while True:
    ret,frame = cap.read()
    
    hsvImage = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    
     
    mask = cv2.inRange(hsvImage,yellow_lower,yellow_upper)
    
    result = cv2.bitwise_and(frame,frame,mask=mask)
    
    cv2.imshow('original',frame)
    cv2.imshow("mask",mask)
    cv2.imshow("yellow detection",result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
```
![Screenshot 2024-12-12 151149](https://github.com/user-attachments/assets/8c9358a6-3d03-47af-80e0-817d8fc090ea)

## Draw bounding Box to the detected color

```python
import cv2
import numpy as np


cap = cv2.VideoCapture(0)


yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 225, 225])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    mask = cv2.inRange(hsvImage, yellow_lower, yellow_upper)
    
    
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    
    
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Yellow Detection', result)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
![Screenshot 2024-12-12 152242](https://github.com/user-attachments/assets/3c2b5ff5-fc57-4ee8-b0b0-d404577b4281)


