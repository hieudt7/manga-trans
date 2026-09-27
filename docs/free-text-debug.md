# Debug FREE_TEXT — City Hunter V01_020

Ảnh kiểm chứng: `test/cityhunter/v6/City hunter V01_020.png`.

## Luồng thực tế

`koharu-pipeline/src/pipeline.rs` gọi `Model::detect` trong `koharu-ml/src/facade.rs`: PPDocLayoutV3 → segmentation → font detection trên ảnh gốc. OCR và balloon detection cập nhật các block. `Model::inpaint` gọi `inpaint_document`, phân loại theo container, rồi chạy nhánh balloon hoặc `Lama::inference_adaptive`. Cuối cùng `Renderer::render` trong `koharu-renderer/src/facade.rs` dàn và vẽ bản dịch.

## Nguyên nhân đã tái hiện

- Segmentation thiếu một phần viền trắng dày. Giãn cố định 3 px vẫn để lại đường viền hình chữ. Một dấu gạch dài bị bỏ sót hoàn toàn.
- Bộ chọn nền chấp nhận tô phẳng khi 60% mẫu gần màu trung vị. Nền có speed line bị tô thành mảng trắng.
- `source_text_rows` dùng mask nét chữ làm vùng dàn chữ trên nền tối. Một đoạn nhiều cột vẫn là một block, nhưng tiếng Việt bị ép vào các khoảng nhỏ của từng nét, còn 12 px.
- Bộ dò khoảng trắng nhận vùng trắng trong speed line là một khung chú thích và thu vùng dàn chữ xuống phần dưới block.
- Nhánh balloon và adaptive nhận chung mask; cửa sổ context của balloon có thể lấy cả chữ không thuộc balloon. Nhánh adaptive còn xoá bookkeeping cho cả context thay vì chỉ pixel đã xử lý.

## Sửa đổi

- `doc.segment` giữ mask segmentation chưa giãn. Nhánh balloon vẫn tạo mask với thứ tự giãn/lọc như trước. Khi mở tài liệu đã lưu mask từ bản cũ, chạy lại Detect trước Inpaint.
- `adaptive_text_masks` tách mask chữ và mask xoá. Khôi phục viền sáng trong phạm vi gần segmentation, bổ sung lỗ nhỏ bị bao kín, và tìm dấu gạch mảnh phù hợp hướng chữ khi OCR có dấu gạch. Giãn chống rìa ảnh từ 1–4 px theo kích thước glyph, không theo chiều rộng cả block nhiều cột.
- Tách quyền sở hữu mask giữa các nhánh. Chỉ ghép pixel trong mask về ảnh gốc. Context lớn không đồng nghĩa vùng được phép thay đổi lớn.
- Bỏ quy tắc đa số 60%. Nền phẳng dùng tái tạo cục bộ; gradient dùng nội suy mặt phẳng; nền hỗn hợp dùng LaMa. Mẫu nền loại trừ chữ của tất cả các block.
- Vùng dàn chữ lấy từ block logic và nền đã phục hồi, không nhận mask nét chữ. Trên nền tối, tìm hình chữ nhật đủ rộng tránh phần hình vẽ sáng; với nền hỗn hợp, giữ bbox logic. Vùng trắng chỉ được coi là container nếu bao phủ phần lớn block gốc.
- Nhánh FREE_TEXT dùng màu chữ dự đoán trên ảnh gốc khi không có style người dùng; giữ cơ chế ưu tiên stroke hiện có.

## Tái hiện

Đây là replay để cô lập redraw/layout: dùng bbox và OCR đã lưu trong `debug-text-type`, bản dịch tiếng Việt cố định và style đen/viền trắng cố định. Không gọi lại OCR hay dịch qua LLM. Các container được dựng từ bbox đã refit trong snapshot; đây không phải một phép chạy lại toàn bộ detector/LLM.

Tạo mask đầu vào nếu chưa có (cần model segmentation đã cache):

```sh
mkdir -p debug-free-text
cargo run --release -p koharu-ml --bin manga-text-segmentation-2025 -- \
  --cpu --input 'test/cityhunter/v6/City hunter V01_020.png' \
  --output debug-free-text/probability.png \
  --mask-output debug-free-text/source-mask-baseline.png
```

Chạy regression và xuất đủ checkpoint:

```sh
KOHARU_FREE_TEXT_OUT=debug-free-text/current \
cargo test --release -p koharu-ml --test free_text --offline -- --ignored --nocapture

cargo test --release -p koharu-ml -p koharu-renderer --lib --offline
```

Trong `debug-free-text/current`:

1. `01_original.png`
2. `02_source_text_mask.png` — mask chữ FREE_TEXT đã khôi phục
3. `03_removal_mask.png` — mask được phép sửa của FREE_TEXT
4. `04_cleaned_before_typesetting.png` — checkpoint chính
5. `05_available_region.png`
6. `06_layout_preview.png`
7. `07_final.png`

Mỗi thư mục con mang ID block cũng chứa đủ 7 ảnh crop để xem chi tiết. `blocks.json` lưu layout/style. `debug-free-text/baseline` chứa kết quả trước sửa để so sánh; không ghi đè thư mục này khi chạy bản đã sửa. Nếu baseline có sẵn, regression kiểm tra từng layer chữ balloon không thay đổi. Mask ở checkpoint 02–03 chỉ dành cho FREE_TEXT; phần balloon phía trên trang được xử lý bằng mask riêng.

Biến `KOHARU_DEBUG_FREE_TEXT=/đường/dẫn/debug` bật log chọn cách phục hồi và xuất checkpoint 01–04 từ pipeline ứng dụng. Khi xử lý nhiều trang, dùng thư mục riêng cho từng lượt để tránh ghi đè.

## Phạm vi kiểm chứng

Replay kiểm tra pixel ngoài removal mask ở toàn panel FREE_TEXT không thay đổi, dấu gạch gốc đã mất, hai block vẫn là hai đơn vị dịch, cỡ chữ tối thiểu 26 px và màu fill được giữ. Trên ảnh này, cỡ chữ thực tế là 30 px ở trái và 33 px ở phải. Các layer chữ balloon được so với baseline theo từng pixel.

Khôi phục viền vẫn là heuristic dựa trên segmentation và tương phản sáng; không thể coi một trang kiểm chứng là bảo đảm đúng mọi kiểu chữ/mọi nền. LaMa suy đoán phần artwork bị chữ che, không phục hồi được ground truth không có trong ảnh. Vùng layout trên nền hỗn hợp hiện dùng bbox logic; các block ôm sát hình vẽ phức tạp cần thêm ảnh regression.
