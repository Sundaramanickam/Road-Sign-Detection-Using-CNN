import argparse
import time
import requests
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize


COUNTER, FPS = 0, 0
START_TIME = time.time()
previous_timestamp_ms = 0

ESP32_CAM_URL = 'http://00:22:57/cam-hi.jpg'

def get_esp32_frame():
    try:
        resp = requests.get(ESP32_CAM_URL, timeout=2)
        img_array = np.frombuffer(resp.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except:
        print("Failed to fetch frame from ESP32-CAM.")
        return None

def run(model: str, max_results: int, score_threshold: float) -> None:
    global previous_timestamp_ms, COUNTER, START_TIME, FPS

    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_result_list = []
    detection_frame = None

    def save_result(result: vision.ObjectDetectorResult, unused_image: mp.Image, timestamp_ms: int):
        global COUNTER, START_TIME, FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        detection_result_list.clear()
        detection_result_list.append(result)
        COUNTER += 1

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)

    while True:
        frame = get_esp32_frame()
        if frame is None:
            continue

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        current_timestamp = int(time.monotonic() * 1000)
        if current_timestamp <= previous_timestamp_ms:
            current_timestamp = previous_timestamp_ms + 1
        previous_timestamp_ms = current_timestamp

        try:
            detector.detect_async(mp_image, current_timestamp)
        except ValueError as e:
            print(f"Detection error: {e}")
            continue

        current_frame = frame.copy()
        fps_text = f'FPS = {FPS:.1f}'
        cv2.putText(current_frame, fps_text, (left_margin, row_size), cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            result = detection_result_list[0]

            
            result.detections = [d for d in result.detections if d.categories[0].score >= 0.91]

            if result.detections:
                current_frame = visualize(current_frame, result)
                detection_frame = current_frame.copy()

            detection_result_list.clear()

        if detection_frame is not None:
            cv2.imshow('ESP32-CAM Object Detection', detection_frame)
        else:
            cv2.imshow('ESP32-CAM Object Detection', current_frame)

        if cv2.waitKey(1) == 27:  
            break

        time.sleep(0.01)

    detector.close()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best.tflite')
    parser.add_argument('--maxResults', default=5, type=int)
    parser.add_argument('--scoreThreshold', default=0.25, type=float)
    args = parser.parse_args()

    run(args.model, args.maxResults, args.scoreThreshold)

if __name__ == '__main__':
    main()
