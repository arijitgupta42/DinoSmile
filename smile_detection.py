import cv2
from pynput.keyboard import Key,Controller
keyboard = Controller()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect(gray, frame):
	flag = False
	face = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(50,50))
	for(x, y, w, h) in face:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 130, 0), 2)
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = frame[y:y + h, x:x + w]
		smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=55, minSize=(35,35))
		for(sx, sy, sw, sh) in smile:
			cv2.rectangle(roi_color, (sx,sy), (sx + sw, sy+sh), (255,0,130),2)
			flag = True

	return frame, flag

def main():
	video_capture = cv2.VideoCapture(0)
	while True:
		#Capture video frame by frame
		_, frame = video_capture.read()

		#Capture image in monochrome
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#Call the detect function
		canvas, flag = detect(gray, frame)

		#Display the result on camera feed
		cv2.imshow('Video', canvas)

		#Press the spacebar if smile detected
		if flag:
			keyboard.press(Key.space)
			keyboard.release(Key.space)		
	    
		#Break control once desired key is pressed
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

