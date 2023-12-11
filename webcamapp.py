import tkinter as tk
from tkinter import filedialog
from threading import Thread
import cv2
import PIL.Image, PIL.ImageTk
from ultralytics import YOLO
import cvzone
import math
import time

class ObjectDetectionApp:
    def __init__(self, root, window_width, window_height):
        self.root = root
        self.root.title("Object Detection App")

        self.window_width = window_width
        self.window_height = window_height

        self.video_source = -1  # For Webcam
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.model = YOLO("../Yolo-Weights/yolov8x.pt")

        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop Detection", command=self.stop_detection)
        self.stop_button.pack(side="left", padx=5)
        self.stop_button["state"] = "disabled"

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=5)

        self.is_detecting = False
        self.update()

    def start_detection(self):
        self.is_detecting = True
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        self.thread = Thread(target=self.detect_objects)
        self.thread.start()

    def stop_detection(self):
        self.is_detecting = False
        self.start_button["state"] = "normal"
        self.stop_button["state"] = "disabled"

    def quit_app(self):
        self.is_detecting = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        self.cap.release()
        self.root.destroy()

    def detect_objects(self):
        while self.is_detecting:
            self.new_frame_time = time.time()
            success, frame = self.cap.read()

            if not success:
                print("Video stream ended or unable to read a frame.")
                self.is_detecting = False
                break

            results = self.model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(frame, f'{self.class_names[cls]} {conf}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1)

            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            print(fps)

            resized_frame = cv2.resize(frame, (self.window_width, self.window_height))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo

    def update(self):
        if self.is_detecting:
            self.root.after(10, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root, window_width=800, window_height=600)
    root.mainloop()
