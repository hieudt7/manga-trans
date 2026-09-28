# FREE_TEXT redraw / lettering — bàn giao

Cập nhật: 28/09/2026. Giao tiếp với người dùng bằng tiếng Việt.

## Cập nhật sửa clipping và spacing

Theo yêu cầu tiếp theo: source hiện dùng `ChalkboardSE-Bold` và
`normalized_translation` để render in hoa; khe dọc giảm xuống **0,3px**.
Dòng rộng hơn logical region trước đây được căn giữa về tọa độ âm và bị
canvas cắt mép (ví dụ chữ C trong “CỦA TÔI RỒI...”). Lettering giờ mở rộng
canvas theo visual bounds có viền, rồi bù tọa độ X khi ghép để giữ trục giữa.
Không đổi wrapping, cỡ chữ hoặc tỷ lệ stroke.

Đã kiểm tra: 88 unit test renderer qua; chạy riêng các test lettering có
font hệ thống đều qua, gồm regression câu in hoa bị tràn và khe dọc 0,3px.
Đã chạy lại pipeline Gemini toàn trang 020 và xem ảnh: chữ C trong “CỦA TÔI RỒI...”
đã đầy đủ. Ảnh: `test/cityhunter/v6/result/City hunter V01_020-Chalkboard-Uppercase-gap03-unclipped.png`.
Dòng cuối vẫn sát mép panel; log vẫn cảnh báo tổng chiều cao vượt logical region.

Đã chạy tiếp pipeline toàn trang 151 cùng cấu hình; ảnh lưu ở
`test/cityhunter/v6/result/City hunter V01_151-Chalkboard-Uppercase-gap03-unclipped.png`.
Tiêu đề đọc rõ; khối chữ nhỏ dưới trái vẫn thành mảng xám khó đọc như vấn đề màu đã ghi nhận.

Phần dưới ghi lại trạng thái và ảnh so sánh **trước lần sửa này**.

## Trạng thái trước lần sửa

**Đang ở bước so sánh và chọn font.** Đã xuất đủ ba phiên bản trang 020:
HL-Comic1, Chalkboard SE Bold chữ thường và Chalkboard SE Bold in hoa.
Người dùng chưa chọn bản cuối. Không tự chuyển sang sửa wrapping hoặc đo stroke nguồn.

- Commit gần nhất: `9bae9da` — `Refine FREE_TEXT comic lettering and remove debug artifacts`.
- Commit này dùng `ChalkboardSE-Bold`, màu nguồn, stroke bằng `font_size × 0.23`, khe dọc 0,5px.
- Working tree hiện đổi riêng font FREE_TEXT sang `HL-Comic1unicode-Normal` để thử; **chưa commit thay đổi font**.
- Code hiện truyền bản dịch giữ nguyên hoa/thường vào `restyle`. HL-Comic1 có hình dáng glyph dạng in hoa.
- Bản Chalkboard in hoa được build tạm với `ChalkboardSE-Bold` và `normalized_translation`, chạy xong rồi khôi phục source về HL-Comic1.
- Vì vậy binary `target/release/koharu` có thể vẫn là bản thử Chalkboard in hoa; **build lại trước khi chạy**, không suy ra cấu hình binary từ source.

## Ảnh để người dùng chọn

Ảnh gốc: `test/cityhunter/v6/City hunter V01_020.png`.
Ảnh mục tiêu: [Target.png](../test/cityhunter/v6/result/Target.png).

| Bản | Ảnh đầy đủ | Crop chi tiết |
|---|---|---|
| HL-Comic1unicode-Normal | [HLComic](<../test/cityhunter/v6/result/City hunter V01_020-HLComic.png>) | [Chi tiết](../test/cityhunter/v6/result/HLComic-detail.png) |
| ChalkboardSE-Bold, giữ hoa/thường | [Chalkboard](<../test/cityhunter/v6/result/City hunter V01_020-Chalkboard.png>) | [Chi tiết](../test/cityhunter/v6/result/Chalkboard-detail.png) |
| ChalkboardSE-Bold, in hoa toàn bộ | [Chalkboard Uppercase](<../test/cityhunter/v6/result/City hunter V01_020-Chalkboard-Uppercase.png>) | [Chi tiết](../test/cityhunter/v6/result/Chalkboard-Uppercase-detail.png) |

Các ảnh nằm cùng folder `test/cityhunter/v6/result/`. Đây là ảnh local trong thư mục bị gitignore, không đi theo commit.
Các bản chạy pipeline Gemini riêng; không mặc định coi chúng là A/B có bản dịch byte-identical.

## Đã làm gì

### 1. Sửa khác biệt màu giữa replay và pipeline

Test tốt ban đầu không chỉ hardcode màu: nó còn gán ruột `[35,35,35]`,
viền `[255,255,255]` và độ dày viền `3px`.
Khi thay cả `FontPrediction` bằng detector thật, cả màu và độ dày cùng đổi,
khiến so sánh bị sai: viền vàng/xám và quá mảnh, đặc biệt bên phải.

Đã thêm lấy màu ruột/viền từ pixel ảnh gốc trong `koharu-ml/src/source_colors.rs`,
trước inpaint. Giữ phân loại container và đường xóa chữ/phục hồi nền hiện có.

### 2. Tách màu khỏi độ dày

Đã thử đo độ dày bằng mask và co theo cỡ chữ, nhưng kết quả không đạt.
Mask segmentation cắt thiếu viền thật, dẫn tới đo thiếu độ dày.
Đã loại bỏ phần đo/co tự động này khỏi đường đang dùng.

Đã kiểm chứng riêng: **màu nguồn + viền cố định 3px**.
Control dựng lại trùng từng pixel với ảnh test tốt; bản đổi màu giữ cùng geometry/alpha.
Đây là bước xác nhận tích hợp màu, **không phải thuật toán viền cuối cùng**.

### 3. Typography theo ảnh Target

Đã triển khai trong `koharu-renderer/src/lettering.rs`:

- Chốt dòng bằng layout cũ trước; sau đó reshape từng dòng bằng face FREE_TEXT.
- Giữ bản dịch, ranh giới dòng và cỡ chữ do layout chọn; không cải tiến thuật toán wrapping.
- Giữ combining marks theo cluster khi dịch chuyển glyph.
- Đo visual bounds bằng alpha raster thực, bao gồm dấu tiếng Việt, AA và viền.
- Điều chỉnh vị trí glyph theo từng cặp, giữ ruột chữ tách nhau nhưng cho viền nối.
- Căn giữa bằng visual bounds, không theo số ký tự.
- Đặt baseline theo biên thực của dòng trước.
- Vẽ stroke glyph bo tròn rồi vẽ fill; không dilation cả dòng thành một khối.

Đã so sánh font cài sẵn. Chalkboard SE Bold có chữ thường thật, weight 700,
hỗ trợ các dấu tiếng Việt đã kiểm tra. Không tải font ngoài.

### 4. Chốt spacing/stroke rồi thử font

Người dùng thấy viền 3px vẫn quá mảnh so với Target, yêu cầu bám sát mẫu.
Đã chuyển sang tham số typography `stroke_width = font_size × 0.23`.
Đây là tỷ lệ style, **không đo viền nguồn tự động**.

Khe dọc lần lượt giảm từ 6–7px xuống 1px, rồi **0,5px** theo yêu cầu.
Người dùng chấp nhận spacing ngang và bản khe dọc 0,5px, yêu cầu commit/dọn debug.
Sau commit mới yêu cầu thử HL-Comic1 và thêm bản Chalkboard in hoa để chọn.

## Thông số đang giữ

| Thông số | Hiện trạng |
|---|---|
| Font trong source hiện tại | `HL-Comic1unicode-Normal` — thay đổi thử, chưa commit |
| Font của commit `9bae9da` | `ChalkboardSE-Bold` |
| Font size regression thường gặp | Trái 30px, phải 35px; phụ thuộc layout/bản dịch, không hardcode theo trang |
| Fill/stroke RGB | Lấy từ nguồn; trang 020: fill `[50,51,44]`, viền trái `[252,253,250]`, phải `[252,252,249]` |
| Stroke width | `font_size × 0.23`; ví dụ trái 6,9px, phải 8,05px |
| Tracking | Theo cặp glyph và chỗ trống; khoảng hở ruột chữ khoảng 1–3px ở cỡ chữ này, không phải global tracking cố định |
| Khe dọc | 0,5px giữa bounds; baseline dùng số thực, rasterizer AA ở vị trí nửa pixel |
| Hoa/thường | Giữ bản dịch khi reshape; bản in hoa là biến thể render thử riêng |

0,5px là khoảng cách hình học. Không đồng nghĩa luôn có một hàng pixel hoàn toàn trong suốt giữa các dòng sau AA.

## File code liên quan

- `koharu-renderer/src/facade.rs`: chọn face, tỷ lệ stroke, gọi post-layout lettering và rasterizer.
- `koharu-renderer/src/lettering.rs`: reshape giữ ranh giới dòng, tracking, visual bounds, baseline, unit test.
- `koharu-renderer/src/lib.rs`: khai báo module lettering.
- `koharu-ml/src/source_colors.rs`: lấy màu nguồn; không sửa khi chỉ thử font/spacing.

## Kiểm tra đã thực hiện

- 90 test renderer qua ở bản khe 0,5px, gồm kiểm tra dấu và giữ dòng khi đổi face.
- Replay trang 020 qua; trong các lần đối chiếu trước dọn debug, mask nguồn,
  removal mask và ảnh sau inpaint trùng từng pixel với baseline.
- Đã chạy pipeline Gemini thật cho trang 020 và xem ảnh đầu ra nhiều lần.
- Trang 151 cũng đã chạy: tiêu đề đọc được; khối chữ nhỏ dưới trái có fill/stroke
  cùng màu nên khó đọc. Chưa xử lý vấn đề đó trong task typography.
- Ba biến thể font gần nhất đã build và chạy pipeline; không coi test cũ là kiểm chứng mọi lựa chọn font mới.

## Giới hạn còn lại

- Wrapping hiện tại chưa tốt; người dùng đã yêu cầu tách riêng, không tự sửa.
- Tăng viền/đổi font làm tổng chiều cao khác đi. Khe 0,5px giúp thu gọn,
  nhưng không đảm bảo mọi bản dịch vừa logical region. Có log cảnh báo overflow.
- Bản Chalkboard in hoa cao hơn, dòng cuối sát mép panel.
- Source dùng font hệ thống. Nếu font không có, code fallback về font hiện có;
  không hứa ảnh giống nhau trên máy chưa cài font đó.

## Cách chạy tiếp

Theo [test-folder-pipeline-headless.md](test-folder-pipeline-headless.md).

1. Build lại: `cargo build --release -p koharu --offline`.
2. Copy riêng ảnh cần test vào folder tạm mới; tránh `result/` cũ làm pipeline skip.
3. Chạy `./target/release/koharu --headless --port=9999`.
4. Mở folder qua `/api/v1/folder/open-path`.
5. POST `/api/v1/jobs/pipeline-folder` với:

```json
{
  "llmModelId": "gemini:gemini-3.1-flash-lite-preview",
  "language": "vi-VN",
  "processWithCharacter": true
}
```

6. Poll `/api/v1/folder/session` tới `hasResult: true`; xem ảnh sau khi hoàn tất.
7. Lưu file theo tên biến thể, không ghi đè ảnh so sánh; dừng server test.

Người dùng đã cho phép chạy Gemini cho các thử nghiệm này.
Đã từng bị kẹt ở macOS Keychain: job báo running nhưng step còn null.
Sau khi người dùng Allow thì job tự tiếp tục; **kiểm tra file result/log trước khi
khởi động lại**, vì từng có lần ảnh đã xuất xong dù lượt trước còn báo chờ.
Không in khóa API ra log hoặc đưa vào tài liệu.

## Dọn dẹp và lưu ý khi tiếp quản

Theo yêu cầu người dùng, đã xóa các ảnh/log debug FREE_TEXT, replay tạm,
example so sánh font và `docs/free-text-debug.md` trong đợt commit `9bae9da`.
Giữ ảnh nguồn, Target, unit test và tài liệu headless.
Ảnh so sánh font ở trên được tạo lại sau đợt dọn và hiện cần giữ để người dùng chọn.
Không dựa vào đường dẫn debug cũ trong lịch sử hội thoại: nhiều file đã bị xóa.

Có thay đổi scan nhân vật/style profile song song trong repository;
không stage, revert hay xóa các file đó khi làm lettering.

## Bước kế tiếp

**Chờ người dùng chọn font và chế độ hoa/thường trong ba ảnh.**
Sau lựa chọn: cập nhật đúng cấu hình, kiểm tra cần thiết và commit khi được yêu cầu.
Chưa triển khai wrapping mới hoặc tự động đo stroke nguồn.

## Kiểm chứng thiếu “Chương 7” trên trang 151

Đã chạy lại pipeline với log tạm tại OCR vision và Gemini translate:
- Vision đọc: `第7話 死亡黑名單之卷`.
- Payload dịch gửi nguyên văn chuỗi trên trong block `[0]`.
- Gemini trả: `Chương 7: Danh sách tử thần`.
- Log tiếp theo: `stripped speaker prefix from translation prefix=Chương 7`.

Nguyên nhân xác nhận tại `koharu-llm/src/facade.rs`, hàm
`strip_speaker_prefix`: nhận nhầm tiền tố tiêu đề là tên người nói rồi cắt.
Chưa sửa trong lượt kiểm chứng này. Log: `/tmp/koharu-151-text-trace.log`.
Đã gỡ instrumentation tạm khỏi source và dừng server; binary release vẫn
là bản có log opt-in `KOHARU_TRACE_TEXT_151` (mặc định tắt), build lại ở lượt sau.

## Sửa hậu xử lý riêng FREE_TEXT

Đã bỏ bước `strip_speaker_prefix` cho block ngoài balloon (theo tâm block
và bbox balloon; block `balloon_fitted` vẫn giữ xử lý balloon).
Áp dụng cho dịch document, batch pipeline, retry từng block và dịch block riêng.
Balloon tiếp tục xử lý tiền tố tên người nói như trước. Không đổi prompt/OCR.
Regression xác nhận `Chương 7: Danh sách tử thần` được giữ đầy đủ ở FREE_TEXT,
còn `RYO: Xin chào` trong balloon vẫn thành `Xin chào`.
`cargo test -p koharu-llm -p koharu-pipeline --lib --offline` qua.
Đã build release và chạy lại pipeline Gemini trang 151: ảnh giữ đầy đủ
`CHƯƠNG 7: DANH SÁCH TỬ THẦN`. Kết quả:
`test/cityhunter/v6/result/City hunter V01_151-Chalkboard-Uppercase-gap03-prefix-fixed.png`.
Chữ nhỏ dưới trái vẫn khó đọc do vấn đề màu đã ghi nhận.
