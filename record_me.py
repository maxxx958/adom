import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
is_recording = False

print("Press 'Space' to start/stop recording. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to exit
        break
    elif key == 32:  # Space key to start/stop recording
        if is_recording:
            is_recording = False
            out.release()
            print("Recording stopped.")
        else:
            is_recording = True
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print("Recording started.")

    if is_recording:
        out.write(frame)

cap.release()
if is_recording:
    out.release()
cv2.destroyAllWindows()
