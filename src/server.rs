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

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<OcrEngine>,
    pub sem: Arc<Semaphore>,
}

pub async fn serve(engine: Arc<OcrEngine>, port: u16) -> anyhow::Result<()> {
    let state = AppState {
        engine,
        sem: Arc::new(Semaphore::new(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        )),
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
async fn ocr_json(State(st): State<AppState>, body: String) -> Response {
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
            let tmp = std::env::temp_dir().join(format!("rustocr-upload-{}.pdf", std::process::id()));
            std::fs::write(&tmp, &bytes).map_err(|e| anyhow::anyhow!("запись временного файла: {}", e))?;
            let dpi = std::env::var("RUSTOCR_DPI")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(crate::pdf::DEFAULT_DPI);
            let t0 = std::time::Instant::now();
            let images = crate::pdf::render_pdf(&tmp, dpi);
            let _ = std::fs::remove_file(&tmp);
            let images = images?;
            let render_ms = t0.elapsed().as_millis() as u64;
            let t1 = std::time::Instant::now();
            let mut pages = Vec::with_capacity(images.len());
            for (n, img) in images.into_iter().enumerate() {
                let r = engine.recognize(&img)?;
                let mut v = r.to_json();
                v["page"] = serde_json::json!(n + 1);
                pages.push(v);
            }
            let ocr_ms = t1.elapsed().as_millis() as u64;
            Ok(serde_json::json!({
                "pages": pages,
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
        let img = image::load_from_memory(&bytes)
            .map_err(|e| anyhow::anyhow!("не удалось декодировать картинку: {}", e))?;
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
