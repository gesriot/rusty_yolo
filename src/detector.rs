//! YOLO-детектор на ONNX Runtime, экспортируемый в Python через PyO3

use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage};
use ndarray::{Array4, ArrayViewD};
use numpy::PyReadonlyArrayDyn;
use once_cell::sync::OnceCell;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub type DetectResult = (f32, f32, f32, f32, f32, String);
pub type Detections = Vec<DetectResult>;

/// Bounding-box + метаданные
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub confidence: f32,
    pub class_id: usize,
    pub class_name: String,
}

/// Классы COCO
const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Обёртка над сессией ORT
pub struct YoloDetector {
    session: Session,
    iw: usize,
    ih: usize,
}

impl YoloDetector {
    pub fn new(model: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model)?;
        Ok(Self {
            session,
            iw: 640,
            ih: 640,
        })
    }

    fn preprocess(&self, img: &DynamicImage) -> Vec<f32> {
        let resized = img.resize_exact(self.iw as u32, self.ih as u32, FilterType::Triangle);
        let rgb = resized.to_rgb8();
        let mut v = Vec::with_capacity(3 * self.iw * self.ih);
        for c in 0..3 {
            for y in 0..self.ih {
                for x in 0..self.iw {
                    v.push(rgb.get_pixel(x as u32, y as u32)[c] as f32 / 255.0);
                }
            }
        }
        v
    }

    fn iou(a: &Detection, b: &Detection) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.w).min(b.x + b.w);
        let y2 = (a.y + a.h).min(b.y + b.h);
        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }
        let inter = (x2 - x1) * (y2 - y1);
        inter / (a.w * a.h + b.w * b.h - inter)
    }

    fn nms(&self, mut dets: Vec<Detection>, thr: f32) -> Vec<Detection> {
        dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut keep = Vec::new();
        let mut sup = vec![false; dets.len()];
        for i in 0..dets.len() {
            if sup[i] {
                continue;
            }
            keep.push(dets[i].clone());
            for j in (i + 1)..dets.len() {
                if sup[j] || dets[i].class_id != dets[j].class_id {
                    continue;
                }
                if Self::iou(&dets[i], &dets[j]) > thr {
                    sup[j] = true
                }
            }
        }
        keep
    }

    fn postprocess(&self, out: ArrayViewD<'_, f32>, ow: f32, oh: f32) -> Result<Vec<Detection>> {
        const N: usize = 8400;
        const C: usize = 80;
        let data = out.as_slice().ok_or_else(|| anyhow!("slice"))?;
        let sx = ow / self.iw as f32;
        let sy = oh / self.ih as f32;
        let mut res = Vec::new();

        for i in 0..N {
            let (cx, cy, w, h) = (data[i], data[N + i], data[2 * N + i], data[3 * N + i]);
            let (mut best_conf, mut best_cls) = (0.0, 0usize);
            for j in 0..C {
                let conf = data[(4 + j) * N + i];
                if conf > best_conf {
                    best_conf = conf;
                    best_cls = j
                }
            }
            if best_conf > 0.5 {
                res.push(Detection {
                    x: (cx - w / 2.0) * sx,
                    y: (cy - h / 2.0) * sy,
                    w: w * sx,
                    h: h * sy,
                    confidence: best_conf,
                    class_id: best_cls,
                    class_name: COCO_CLASSES[best_cls].to_string(),
                });
            }
        }
        Ok(res)
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<Detection>> {
        let (ow, oh) = (img.width() as f32, img.height() as f32);
        let input = self.preprocess(img);
        let arr = Array4::from_shape_vec((1, 3, self.ih, self.iw), input)?;
        let tensor = Tensor::from_array(arr)?;
        let outputs = self.session.run(ort::inputs![tensor])?;
        let (_name, value) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("пустой вывод"))?;
        let out = value.try_extract_array::<f32>()?;
        let dets = self.postprocess(out, ow, oh)?;
        Ok(self.nms(dets, 0.45))
    }
}

// Глобальный singleton детектора
static DETECTOR: OnceCell<Mutex<YoloDetector>> = OnceCell::new();

fn get_detector() -> Result<&'static Mutex<YoloDetector>> {
    DETECTOR.get_or_try_init(|| {
        let model_path =
            std::env::var("YOLO_MODEL").unwrap_or_else(|_| "models/yolov8n.onnx".to_string());
        Ok(Mutex::new(YoloDetector::new(&model_path)?))
    })
}

/// PyO3-функция: release GIL перед инференсом
#[pyfunction]
fn detect(py: Python, frame: &PyAny) -> PyResult<Detections> {
    // 1) извлечение массива
    let array: PyReadonlyArrayDyn<u8> = frame.extract()?;
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("Ожидается H×W×3 RGB uint8"));
    }
    let (h, w) = (shape[0] as u32, shape[1] as u32);
    if h != 640 || w != 640 {
        return Err(PyValueError::new_err(format!(
            "Кадр должен быть 640×640, получено {}×{}",
            w, h
        )));
    }

    let buf = array.as_slice()?.to_vec();
    let rgb = image::RgbImage::from_raw(w, h, buf)
        .ok_or_else(|| PyRuntimeError::new_err("from_raw failed"))?;
    let img = DynamicImage::ImageRgb8(rgb);

    let det_mutex = get_detector().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // 2) release GIL при инференсе
    let dets_res = py.allow_threads(|| {
        let mut det = det_mutex.lock().unwrap();
        det.detect(&img)
    });

    // 3) обработка результата
    let dets = dets_res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(dets
        .into_iter()
        .map(|d| (d.x, d.y, d.w, d.h, d.confidence, d.class_name))
        .collect())
}

/// Регистрация модуля
pub fn init_python_module(py: Python) -> PyResult<()> {
    let m = PyModule::new(py, "rusty_yolo")?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("rusty_yolo", m)?;
    Ok(())
}
