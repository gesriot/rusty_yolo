"""
Producer–Consumer для capture → inference → draw
"""
import cv2
import threading
import queue
import numpy as np
import rusty_yolo

# Очереди для кадров и результатов
frame_queue = queue.Queue(maxsize=1)
det_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()


def inference_worker():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            detections = rusty_yolo.detect(frame)
        except Exception as e:
            print(f"[ERROR][Inference] {e}")
            detections = []
        # Кладём последний результат
        if not det_queue.full():
            det_queue.put(detections)
        else:
            _ = det_queue.get_nowait()
            det_queue.put(detections)


def run(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # Запуск потока инференса
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    print(f"[Python] ▶ Начало воспроизведения: {video_path}")
    import time
    start_time = time.time()
    frame_count = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_count += 1

        # Отправляем последний кадр на инференс (non-blocking)
        if not frame_queue.full():
            frame_queue.put(frame_bgr)
        else:
            _ = frame_queue.get_nowait()
            frame_queue.put(frame_bgr)

        # Получаем последний результат, если есть
        try:
            detections = det_queue.get_nowait()
        except queue.Empty:
            detections = []

        # Рисуем bbox'ы
        draw = frame_bgr.copy()
        h0, w0 = draw.shape[:2]
        for x, y, w, h, conf, cls in detections:
            # относительные → абсолютные
            if max(x, y, w, h) <= 1.0:
                x *= w0; y *= h0; w *= w0; h *= h0
            x1, y1 = map(int, (round(x), round(y)))
            x2, y2 = map(int, (round(x + w), round(y + h)))
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(draw, f"{cls} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Вывод FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed>0 else 0
        cv2.putText(draw, f"FPS: {fps:.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("YOLO Realtime", draw)
        if cv2.waitKey(1) & 0xFF in (ord('q'),27):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[Python] ■ Стоп. Обработано кадров: {frame_count}")