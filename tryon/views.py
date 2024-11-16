from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import os

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def overlay_sunglasses(frame, sunglasses_image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        overlay_resized = cv2.resize(sunglasses_image, (w, int(w * sunglasses_image.shape[0] / sunglasses_image.shape[1])))

        y_offset = y + int(h / 1000)
        x_offset = x

        y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + overlay_resized.shape[0])
        x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + overlay_resized.shape[1])

        overlay_height = y2 - y1
        overlay_width = x2 - x1

        overlay_resized = overlay_resized[:overlay_height, :overlay_width]

        alpha_s = overlay_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (
                alpha_s * overlay_resized[:, :, c] +
                alpha_l * frame[y1:y2, x1:x2, c]
            )

    return frame

def video_stream(selected_sunglasses):
    cap = cv2.VideoCapture(0)
    sunglasses_image = cv2.imread(f'static/tryon/{selected_sunglasses}.png', -1)  # Load the selected sunglasses image
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = overlay_sunglasses(frame, sunglasses_image)
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
    cap.release()

def stream_view(request):
    selected_sunglasses = request.GET.get('sunglasses', 'sunglasses1')  # Default to 'sunglasses1' if none is selected
    return StreamingHttpResponse(video_stream(selected_sunglasses), content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, 'tryon/home.html')
