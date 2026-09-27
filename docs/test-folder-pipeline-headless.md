# Test pipeline folder qua server headless

Cách chạy pipeline folder (detect → OCR → balloon → dịch LLM → inpaint →
render) trên một vài trang cụ thể mà không cần mở GUI/Tauri. Dùng khi cần
kiểm tra một thay đổi ở detection/inpaint/render trên ảnh thật, có dịch LLM
đàng hoàng (khác với `translate_and_render_sample`/`render_layout_sample`
trong `koharu-pipeline` — những test đó chạy 1 trang qua `cargo test`, không
qua HTTP API và không mô phỏng đúng luồng Folder mode).

## 1. Build lại binary release

```sh
cargo build --release -p koharu
```

Luôn build lại trước khi test — nếu sửa code mà không build, server headless
sẽ chạy binary cũ. Kiểm tra nhanh bằng cách so sánh timestamp của binary
(`ls -la target/release/koharu`) với các file vừa sửa, hoặc cứ build lại cho
chắc.

## 2. Gom các trang cần test vào một thư mục riêng

Pipeline folder xử lý **toàn bộ** ảnh trong thư mục nó trỏ tới, và bỏ qua ảnh
đã có sẵn kết quả trong `result/`. Nếu trỏ thẳng vào một thư mục test đang có
nhiều trang khác (vd `test/cityhunter/v6`), nó sẽ chạy luôn cả những trang
không liên quan — tốn API call vô ích. Copy riêng các trang cần test ra một
thư mục tạm:

```sh
mkdir -p /tmp/redraw-test
cp "test/cityhunter/v6/City hunter V01_020.png" \
   "test/cityhunter/v6/City hunter V01_151.png" \
   /tmp/redraw-test/
```

## 3. Chạy server headless

```sh
./target/release/koharu --headless --port=9999 > /tmp/koharu-server.log 2>&1 &
```

`--headless` chỉ bật HTTP server (`koharu-rpc`) + load model ML/LLM, không mở
webview. Đợi vài giây tới khi `/meta` trả 200:

```sh
curl -s http://127.0.0.1:9999/api/v1/meta
```

Lưu ý mọi route đều có prefix `/api/v1` (xem `koharu-rpc/src/server.rs`,
`.nest("/api/v1", ...)`) — gọi thẳng `/folder/open-path` sẽ ra 404.

Trên máy không có CUDA (macOS chẳng hạn), log có thể báo
`CUDA and Metal are not available. Using CPU device.` — chậm hơn nhưng vẫn
chạy được, chỉ 1-2 trang thì chấp nhận được.

## 4. Mở folder session + chạy job

```sh
curl -X POST http://127.0.0.1:9999/api/v1/folder/open-path \
  -H "Content-Type: application/json" \
  -d '{"path":"/tmp/redraw-test"}'

curl -X POST http://127.0.0.1:9999/api/v1/jobs/pipeline-folder \
  -H "Content-Type: application/json" \
  -d '{
    "llmModelId": "gemini:gemini-3.1-flash-lite-preview",
    "language": "vi-VN",
    "processWithCharacter": true
  }'
```

- `llmModelId` dạng `<provider>:<model>`. Dùng `gemini:gemini-3.1-flash-lite-preview`
  cho test nhanh/rẻ — không cần truyền `llmApiKey`, provider gemini tự lấy
  key trong `gemini_keys.txt` ở gốc repo (xem
  `koharu-llm/src/providers/mod.rs`, `key_pool::collect_keys`).
- `processWithCharacter: true` nếu muốn pipeline dùng character library đã
  scan sẵn cho bộ truyện đó (vd City Hunter) để có context nhân vật/xưng hô.

## 5. Poll tới khi xong

```sh
curl -s http://127.0.0.1:9999/api/v1/folder/session
```

Đợi tới khi tất cả file trong `files[]` có `hasResult: true`. Kết quả được
lưu trực tiếp vào `<thư mục tạm>/result/<tên file gốc>.png`.

## 6. Xem kết quả

Ảnh gốc thường to (vd 1760x2500) — nên resize trước khi xem để đỡ tốn
context:

```sh
sips -Z 900 "/tmp/redraw-test/result/City hunter V01_020.png" \
  --out /tmp/preview_020.png
```

Copy kết quả vào chỗ cần lưu (vd `test/cityhunter/v6/result/`) nếu muốn giữ
lại để so sánh về sau.

## 7. Dọn dẹp

```sh
kill %1   # hoặc kill <pid> của tiến trình koharu --headless
rm -rf /tmp/redraw-test
```

## Việc cần làm thêm (chưa làm trong lần chạy đầu)

- Muốn xem log chi tiết chọn phương án phục hồi nền (flat/gradient/LaMa) thì
  set biến môi trường `KOHARU_DEBUG_FREE_TEXT=1` trước khi chạy server —
  xem `restore_region: plan chosen` trong `koharu-ml/src/lama/mod.rs:663`.
- Máy có Apple Silicon (M-series) nhưng log báo không dùng được Metal cho ML
  model — chưa rõ nguyên nhân, nếu cần tốc độ thì nên tìm hiểu vì sao
  `device()` không pick Metal.
