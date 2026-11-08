//! HTTP-сервер (Axum): /ocr, /ocr/json, /health, демо-страница на /.

use crate::pipeline::OcrEngine;
use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Semaphore;

const MAX_BODY: usize = 64 * 1024 * 1024; // 64 МБ

/// Верхняя граница одновременных OCR-задач (env RUSTOCR_MAX_CONC, 1..=64).
fn max_concurrency() -> usize {
    std::env::var("RUSTOCR_MAX_CONC")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
        .clamp(1, 64)
}

/// Проверить API-ключ: сервер требует его только если задан RUSTOCR_API_KEY.
fn check_api_key(headers: &axum::http::HeaderMap) -> bool {
    let key = match std::env::var("RUSTOCR_API_KEY") {
        Ok(k) => k,
        Err(_) => return true, // ключ не задан — аутентификация выключена
    };
    if key.is_empty() {
        return true;
    }
    match headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        Some(got) => eq_const_time(got.as_bytes(), key.as_bytes()),
        None => false,
    }
}

/// Сравнение без досрочного выхода (не отдаёт длину/префикс по времени).
fn eq_const_time(a: &[u8], b: &[u8]) -> bool {
    let mut diff: u64 = (a.len() ^ b.len()) as u64;
    for i in 0..a.len().max(b.len()) {
        let x = *a.get(i).unwrap_or(&0);
        let y = *b.get(i).unwrap_or(&0);
        diff |= (x ^ y) as u64;
    }
    diff == 0
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<OcrEngine>,
    pub sem: Arc<Semaphore>,
}

pub async fn serve(engine: Arc<OcrEngine>, port: u16) -> anyhow::Result<()> {
    let conc = max_concurrency();
    eprintln!("Одновременных OCR-задач: {} (RUSTOCR_MAX_CONC)", conc);
    let state = AppState {
        engine,
        sem: Arc::new(Semaphore::new(conc)),
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ocr", post(ocr_any))
        .route("/ocr/json", post(ocr_json))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(state);
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("RustOCR слушает http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "engine": "PP-OCRv5 (ONNX Runtime) на Rust",
        "langs": ["ru", "en"],
    }))
}

/// POST /ocr — сырые байты картинки ИЛИ multipart/form-data с полем file.
async fn ocr_any(State(st): State<AppState>, req: Request) -> Response {
    if !check_api_key(req.headers()) {
        return err(
            StatusCode::UNAUTHORIZED,
            "требуется заголовок X-API-Key".to_string(),
        );
    }
    let ct = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = if ct.starts_with("multipart/form-data") {
        let mut mp = match Multipart::from_request(req, &()).await {
            Ok(m) => m,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("multipart: {}", e)),
        };
        let mut found: Option<Bytes> = None;
        loop {
            match mp.next_field().await {
                Ok(Some(field)) => {
                    let name = field.name().unwrap_or("").to_string();
                    let bytes = match field.bytes().await {
                        Ok(b) => b,
                        Err(e) => {
                            return err(StatusCode::BAD_REQUEST, format!("поле {}: {}", name, e))
                        }
                    };
                    if !bytes.is_empty() && (found.is_none() || name == "file" || name == "image")
                    {
                        found = Some(bytes);
                    }
                }
                Ok(None) => break,
                Err(e) => return err(StatusCode::BAD_REQUEST, format!("multipart: {}", e)),
            }
        }
        match found {
            Some(b) => b,
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "в multipart-запросе нет файла".to_string(),
                )
            }
        }
    } else {
        match axum::body::to_bytes(req.into_body(), MAX_BODY).await {
            Ok(b) => b,
            Err(e) => return err(StatusCode::PAYLOAD_TOO_LARGE, format!("тело: {}", e)),
        }
    };
    if bytes.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "пустое тело запроса — пришлите картинку (PNG/JPEG)".to_string(),
        );
    }
    process_image(st, bytes).await
}

#[derive(Deserialize)]
struct Base64Req {
    image: String,
}

/// POST /ocr/json — {"image": "<base64 или data-url>"}.
async fn ocr_json(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    body: String,
) -> Response {
    if !check_api_key(&headers) {
        return err(
            StatusCode::UNAUTHORIZED,
            "требуется заголовок X-API-Key".to_string(),
        );
    }
    let req: Base64Req = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("JSON: {}", e)),
    };
    let mut b64 = req.image.trim().to_string();
    if let Some(pos) = b64.find("base64,") {
        b64 = b64[pos + "base64,".len()..].to_string();
    }
    let bytes = match STANDARD.decode(b64.as_bytes()) {
        Ok(b) => Bytes::from(b),
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("base64: {}", e)),
    };
    process_image(st, bytes).await
}

async fn process_image(st: AppState, bytes: Bytes) -> Response {
    let engine = st.engine.clone();
    let permit = match st.sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => return err(StatusCode::INTERNAL_SERVER_ERROR, "semaphore".into()),
    };
    // PDF? → растеризуем и читаем страницу за страницей
    if crate::pdf::is_pdf_bytes(&bytes) {
        let res = tokio::task::spawn_blocking(move || -> anyhow::Result<serde_json::Value> {
            // уникальная временная папка на запрос: параллельные загрузки
            // не сталкиваются, уборка гарантирована при любом исходе
            let job = crate::pdf::PdfRender::new()?;
            let pdf_path = job.pdf_path();
            std::fs::write(&pdf_path, &bytes)
                .map_err(|e| anyhow::anyhow!("запись временного файла: {}", e))?;
            // число страниц — до рендера, чтобы сразу отвергнуть гиганта
            let n = crate::pdf::page_count(&pdf_path)?;
            if n > crate::pdf::MAX_PAGES {
                anyhow::bail!("в PDF {} страниц — максимум {}", n, crate::pdf::MAX_PAGES);
            }
            let dpi = crate::pdf::dpi_from_env();
            let t0 = std::time::Instant::now();
            let t1 = t0;
            let mut pages = Vec::with_capacity(n);
            let mut skipped = 0usize;
            // рендер порциями: память не растёт с толщиной PDF
            for first in (1..=n as u32).step_by(crate::pdf::CHUNK_PAGES) {
                let last = (first + crate::pdf::CHUNK_PAGES as u32 - 1).min(n as u32);
                let images = job.render_range(&pdf_path, first, last, dpi)?;
                for (k, img) in images.into_iter().enumerate() {
                    let page_no = first as usize + k;
                    let r = engine.recognize(&img)?;
                    skipped += r.skipped;
                    let mut v = r.to_json();
                    v["page"] = serde_json::json!(page_no);
                    pages.push(v);
                }
            }
            let render_ms = t0.elapsed().as_millis() as u64;
            let ocr_ms = t1.elapsed().as_millis() as u64;
            let text = pages
                .iter()
                .map(|p| p["text"].as_str().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n\n");
            Ok(serde_json::json!({
                "pages": pages,
                "text": text,
                "skipped_fragments": skipped,
                "timing_ms": {"render": render_ms, "ocr": ocr_ms,
                                 "total": render_ms + ocr_ms},
            }))
        })
        .await;
        drop(permit);
        return match res {
            Ok(Ok(v)) => Json(v).into_response(),
            Ok(Err(e)) => err(StatusCode::UNPROCESSABLE_ENTITY, format!("{}", e)),
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {}", e)),
        };
    }
    let res = tokio::task::spawn_blocking(move || -> anyhow::Result<crate::pipeline::OcrResult> {
        // лимиты декодирования — защита от «декомпрессионной бомбы»
        let img = crate::imgio::decode_bytes(&bytes)?;
        engine.recognize(&img.to_rgb8())
    })
    .await;
    drop(permit);
    match res {
        Ok(Ok(r)) => Json(r.to_json()).into_response(),
        Ok(Err(e)) => err(StatusCode::UNPROCESSABLE_ENTITY, format!("{}", e)),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("worker: {}", e)),
    }
}

fn err(status: StatusCode, msg: String) -> Response {
    (status, Json(json!({"error": msg}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_time_eq() {
        assert!(eq_const_time("секрет".as_bytes(), "секрет".as_bytes()));
        assert!(!eq_const_time("секрет".as_bytes(), "секреT".as_bytes()));
        assert!(!eq_const_time(b"abc", b"abcd"));
        assert!(!eq_const_time(b"abcd", b"abc"));
        assert!(eq_const_time(b"", b""));
    }

    #[test]
    fn concurrency_in_range() {
        let v = max_concurrency();
        assert!((1..=64).contains(&v), "RUSTOCR_MAX_CONC вне диапазона: {}", v);
    }
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RustOCR — RU+EN</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:920px;margin:32px auto;padding:0 16px;color:#1a1a2e}
 h1{margin-bottom:4px} p.sub{color:#666;margin-top:0}
 #drop{border:2px dashed #aab;padding:28px;border-radius:12px;text-align:center;color:#556;
       cursor:pointer;margin:18px 0;transition:.15s}
 #drop.hover{border-color:#2b6cb0;background:#ebf4ff}
 canvas{max-width:100%;border:1px solid #ccc;border-radius:8px;margin:14px 0;display:none}
 pre{background:#f4f5f7;padding:14px;border-radius:8px;white-space:pre-wrap;min-height:20px}
 .ms{color:#888;font-size:13px;margin-top:-8px}
 code{background:#f4f5f7;padding:2px 6px;border-radius:6px;font-size:13px}
</style></head><body>
<h1>⚡ RustOCR</h1>
<p class="sub">Русский + английский, PP-OCRv5 на Rust. Перетащи картинку или кликни по рамке.</p>
<div id="drop">📷 PNG / JPEG / PDF — брось сюда или выбери файл<input type="file" id="f" hidden accept="image/*,application/pdf"></div>
<canvas id="cv"></canvas>
<p class="ms" id="ms"></p>
<pre id="out">Здесь появится распознанный текст…</pre>
<p>Сервис: <code>POST /ocr</code> · <code>POST /ocr/json</code> · <code>GET /health</code></p>
<script>
const drop=document.getElementById('drop'),f=document.getElementById('f');
drop.onclick=()=>f.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hover')};
drop.ondragleave=()=>drop.classList.remove('hover');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hover');go(e.dataTransfer.files[0])};
f.onchange=()=>f.files[0]&&go(f.files[0]);
async function go(file){
 document.getElementById('out').textContent='Распознаю…';
 const buf=await file.arrayBuffer();
 try{
  const r=await fetch('/ocr',{method:'POST',body:buf,
    headers:{'Content-Type':file.type||'application/octet-stream'}});
  const j=await r.json();
  if(!r.ok){document.getElementById('out').textContent='Ошибка: '+(j.error||r.status);return}
  if(j.pages){ // PDF: несколько страниц
   document.getElementById('out').textContent=j.pages.map(p=>
    '=== Страница '+p.page+' ===\n'+(p.text||'(текст не найден)')).join('\n');
   document.getElementById('ms').textContent=
    'страниц: '+j.pages.length+' · рендер '+j.timing_ms.render+' мс · OCR '+j.timing_ms.ocr+' мс';
   document.getElementById('cv').style.display='none';
   return;
  }
  document.getElementById('out').textContent=j.text||'(текст не найден)';
  document.getElementById('ms').textContent=
    'детекция '+j.timing_ms.det+' мс · чтение '+j.timing_ms.rec+' мс · всего '+j.timing_ms.total+' мс · блоков: '+j.blocks.length;
  const img=new Image();
  img.onload=()=>{
   const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
   cv.style.display='block';cv.width=img.width;cv.height=img.height;
   ctx.drawImage(img,0,0);
   ctx.lineWidth=Math.max(2,img.width/300);
   for(const b of j.blocks){
    ctx.strokeStyle=b.confidence>0.9?'rgba(43,196,70,.95)':'rgba(230,160,30,.95)';
    ctx.strokeRect(b.box[0],b.box[1],b.box[2],b.box[3]);
   }
  };
  img.src=URL.createObjectURL(file);
 }catch(e){document.getElementById('out').textContent='Ошибка сети: '+e}
}
</script></body></html>"#;
