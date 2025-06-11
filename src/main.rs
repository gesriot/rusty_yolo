//! Rust управляет Python-циклом OpenCV

mod detector;

use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn main() -> Result<()> {
    let video_path = std::env::args().nth(1).ok_or_else(|| {
        anyhow!("Использование:\n    cargo run --release -- <путь\\к\\video.mp4>")
    })?;

    if !std::path::Path::new(&video_path).exists() {
        return Err(anyhow!("Файл не найден: {}", video_path));
    }

    //──────────── Python + OpenCV цикл ──────────
    Python::with_gil(|py| -> Result<()> {
        // Регистрируем Rust-модуль
        detector::init_python_module(py)?;

        // Загружаем embedded-скрипт video_io.py
        let code = include_str!("py/video_io.py");
        let video_io = PyModule::from_code(py, code, "video_io.py", "video_io")
            .map_err(|e| anyhow!("Не удалось создать Python-модуль: {}", e))?;

        video_io
            .getattr("run")?
            .call1((video_path,))
            .map_err(|e| anyhow!("Ошибка внутри Python-кода: {}", e))?;
        Ok(())
    })
}
