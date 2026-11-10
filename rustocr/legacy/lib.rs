//! rustocr — скоростной OCR-движок на чистом Rust.
//!
//! Классический пайплайн (по духу — старый Tesseract, но без C++):
//! адаптивная бинаризация -> связные компоненты -> строки -> сегментация
//! глифов -> классификация по шаблонам из встроенных шрифтов.

pub mod cc;
pub mod classify;
pub mod features;
pub mod gray;
pub mod json;
pub mod layout;
pub mod ocr;
pub mod render;
pub mod segment;
pub mod server;
pub mod skew;
pub mod templates;

pub use ocr::{CharOut, Engine, LineOut, Options, Page, WordOut};
pub use templates::Lang;
