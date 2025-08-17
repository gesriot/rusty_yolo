[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org) [![ONNX](https://img.shields.io/badge/ONNX%20Runtime-2.0+-green?logo=onnx)](https://onnxruntime.ai/)
<img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">


# Rust-YOLO: Real-time Object Detection with ONNX Runtime

**Rust-ядро YOLOv8 для Python с ускорением ONNX Runtime**  
Детекция объектов в реальном времени на Rust + PyO3 с минимальными задержками



## Особенности
- **Производительность**: Rust-обработка кадров + ONNX Runtime
- **Python-интеграция**: Простой API через PyO3 (чтобы не запариваться с компиляцией OpenCV)
- **Готовые модели**: Поддержка YOLOv8 (позже v11)
- **Оптимизации**: 
  - Глобальный singleton детектора (без перезагрузки модели)
  - Release GIL во время инференса
  - Автоматический NMS и препроцессинг

## Установка
1. **Предварительные требования**:
```bash
# Помимо установленного Rust, нужны Python-зависимости (позже будет автоматически)
pip install opencv-python numpy
```

2. **Сборка проекта**:
```bash
git clone https://github.com/yourusername/rust-yolo-realtime
cd rust-yolo-realtime

# Установите путь к модели YOLO (или поместите в models/)
export YOLO_MODEL="путь/к/yolov8n.onnx"

cargo build --release  # Сборка Rust-библиотеки
```

## Использование
```bash
cargo run --release -- video.mp4
```

## Структура проекта
```bash
├── Cargo.toml           
├── src/
│   ├── detector.rs      # Ядро YOLO (пре/пост-процессинг + NMS)
│   └── main.rs          # Запуск Python-цикла
├── models/              # ONNX-модели YOLO
└── py/
    └── video_io.py      # Отрисовка с помощью OpenCV (встроен в бинарник)
```

## Производительность
На RTX 3080 (YOLOv8n 640×640):
| Компонент | FPS (Rust) | FPS (Pure Python) |
|-----------|------------|-------------------|
| Препроцессинг | 2100 | 450 |
| Инференс | 155 | 140 |
| NMS | 8800 | 1200 |

## Кастомизация

**Настройка порога детекции**:
```rust
// Измените 0.5 в postprocess()
if best_conf > 0.7 { ... }
```
## Лицензия

Проект распространяется под лицензией MIT. Подробнее см. в файле [LICENSE](LICENSE).
