# City Hunter — trạng thái đọc /style-read (raw-only)

Cập nhật: 2026-09-24. **Tập 1 (Deluxe Edition V01) đã đọc hết và publish
(220/220 trang).** Đây là lần đọc đầu tiên của bộ này — chưa có bản dịch tiếng
Việt, mọi tên riêng/xưng hô trong profile đều là **gợi ý của reader**, không
phải bản dịch đã xuất bản.

**Lưu ý quan trọng về nguồn "raw":** bản scan Deluxe Edition này đã bị rebubble
lời thoại sang **tiếng Trung phồn thể**, không phải tiếng Nhật gốc. Theo đúng
yêu cầu ban đầu, quy trình đọc coi hẳn tiếng Trung này là "raw" — đọc, trích
dẫn, phiên âm y như đang đọc tiếng Nhật, không dịch nó ra tiếng Việt trong ghi
chú. Chữ Nhật gốc (biển hiệu, SFX, ngày tháng Chiêu Hoà) vẫn còn giữ nguyên ở
hậu cảnh và không bị đụng tới.

| Tập | Trang | Trạng thái |
|---|---|---|
| Deluxe Edition V01 | 220/220 | published |

Cast: 46 nhân vật ghi nhận, 3 settled. Profile (đã publish): **7 nhân vật**
(rule >10 trang — bộ mới đọc 1 tập nên nhánh "≥2 tập" của rule chưa có tác
dụng): `ryo, makimura, trum-angel-dust, chi-gai-yuko, inagaki-kouji, iwasaki,
vo-si-nan-nhan`. Face: **27 ảnh**, 6/7 nhân vật đã đủ 4/4, còn
**`vo-si-nan-nhan` ở mức 3/4** — xem "Câu hỏi còn treo" trước khi chạy thêm
face-hunt cho nhân vật này.

## Bug đã sửa trong đợt này — ảnh hưởng mọi series có trang ghép đôi

`digest.py` (`volume_spread`) và `write_profile.py` (`volumes_seen`) tính số
tập một nhân vật xuất hiện bằng cách tách id trang ở dấu `-` **cuối cùng**
(`p.rsplit("-", 1)[0]`). Giả định đó đúng với id kiểu `v01-0353`, nhưng vỡ với
một trang ghép đôi (spread 2 trang gộp thành 1 ảnh) có id kiểu
`v01-CityhunterV01_210-211` — dấu `-` cuối cùng nằm *trong số trang*, không phải
ở ranh giới tập, nên trang đó bị hiểu nhầm thành "tập riêng" của chính nó. Hậu
quả: bất kỳ nhân vật nào xuất hiện trên trang ghép đôi đó cùng với 1 trang khác
sẽ bị tính `volumes_seen = 2`, đủ điều kiện vào cast dù thực tế trang quá ít.
Đã bắt được vì `write_profile.py` add nhầm `quan-ly-casino` (7 trang) và
`can-bo-yunio` (5 trang) vào danh sách "phải thêm" thay vì đúng specc.

**Đã sửa cả hai file**: tách ở dấu `-` **đầu tiên** thay vì cuối cùng
(`p.split("-", 1)[0]`) — prefix tập (`v01`) không bao giờ chứa `-`, số trang thì
có thể (trang ghép đôi). Việc sửa này áp dụng cho **mọi series**, không riêng
City Hunter — nếu Kinnikuman hay series sau này có trang ghép đôi, bug cũ sẽ
tái diễn nếu ai đó revert lại `rsplit`.

Gemini vẫn tự ý thêm 2 nhân vật không đủ điều kiện khác (`chu-boutique` 9
trang, `con-tin` 8 trang — cả hai đều <10 trang, không liên quan gì tới bug
trên) — đã xoá thủ công khỏi `profile.json` cùng các relation lủng lẳng trỏ
tới chúng trong `ryo`/`makimura`, đúng theo tiền lệ xử lý ở Kinnikuman (soát
bằng tay, không prompt lại Gemini).

## Hai lần gộp id thủ công trong tập này

Reader được dặn rõ **không tự gộp id nghi ngờ trùng nhau** — chỉ báo lại, để
người chạy (tôi) tự xem ảnh/lời thoại rồi quyết định. Cả hai lần dưới đây đều
làm theo đúng quy trình: đọc note gốc xác nhận bằng mắt, sửa `cast.json` +
đổi tên id trong các file note liên quan (không chỉ xoá 1 chỗ), rồi chạy lại
`progress.py` để tính lại `pages`/`pairPages` cho đúng.

1. **`co-gai-hieu-lam` (tr.112–125, bị `chu-boutique` bắt cóc) = `makimura`.**
   Bằng chứng: tự xưng "槙村香" (Makimura Kaori/Hương) và nhận là em gái cộng sự
   cũ của Ryo ở tr.126 — trùng khớp `makimura` (trước đó chỉ nghe qua điện
   thoại tr.013, chưa có hình). Ngoại hình mô tả cũ (tóc dài, cardigan cam,
   sống trong căn hộ — suy đoán từ giọng nói qua điện thoại) khác với ngoại
   hình thật khi xuất hiện trực tiếp (tóc xoăn ngang vai) — đã cập nhật field
   `looks` theo lần thấy trực tiếp vì đáng tin hơn suy đoán từ giọng nói.
2. **`co-gai-cong-vien` (tr.201–207, vụ ga-thuoc-no doạ nổ bó que ở công viên)
   = `makimura`.** Bằng chứng: cô gọi Ryo là "阿獠" (xưng hô riêng đã settled
   của makimura↔ryo, tr.204), và chính Ryo gọi thẳng cô là "香"/Hương (tr.207).

**Bẫy kỹ thuật phát hiện khi gộp:** `progress.py` tính lại `pages`/`pairPages`
mỗi lần chạy bằng cách quét dòng `Nhân vật:` và `ADDRESS` trong TOÀN BỘ note —
**gộp thêm (union), không bao giờ xoá bớt** trường cũ trong `cast.json`. Chỉ
đổi tên id trong note KHÔNG đủ để dọn sạch — nếu id cũ đã từng được ghi vào
`cast.json` qua một khối `### Cast` JSON đã "applied" (đánh dấu trong
`cast_applied.json`), nó sẽ **tự tái sinh** ở lần chạy sau vì khối JSON đó
không được fold lại lần hai, nhưng vẫn đứng độc lập trong `characters[]`. Cách
dọn đúng: xoá thẳng entry id cũ khỏi `cast.json` **sau khi** đã đổi tên id
trong mọi note liên quan. Cũng cẩn thận field `hold` khi gộp bằng tay:
`merge_character()` không loại `hold` khỏi danh sách field được gộp, nên nếu id
tạm có `"hold": true` (đúng — nó *nên* hold vì danh tính chưa chắc), gộp thô sẽ
làm `hold: true` "dính" sang cả `makimura" — biến một nhân vật chính đã settled
thành unsettled. Đã gặp lỗi này ở lần gộp thứ 2, phải xoá `hold` khỏi
`makimura` bằng tay sau khi gộp.

## Face-hunt: 3 đợt, tự soát từng ảnh — không nhìn theo sheet

7 nhân vật published, ban đầu 2 ảnh có sẵn (ryo, makimura đã đủ từ lúc đọc),
5 nhân vật còn lại thiếu ảnh. Chạy 3 đợt face-hunt (đợt 1: 5 nhân vật; đợt 2:
sửa 3 ảnh bị đợt 1 làm sai; đợt 3: 4 nhân vật còn thiếu ở mức 2/4). Hai lỗi kỹ
thuật lặp lại nhiều lần đáng ghi nhớ cho lần sau:

- **Box quá lớn bị `usable_face()` âm thầm từ chối** — giới hạn diện tích
  (`w×h` theo tỉ lệ ảnh) là `FACE_MAX_AREA = 0.055`. Reader ước lượng bằng mắt
  dễ vẽ box rộng hơn mức đó (bắt luôn tóc/vai/nền).
- **Box thứ 2 trên CÙNG một trang đã có face bị coi là trùng lặp và bỏ qua
  im lặng** — dedup key chỉ là `(page, side)`, không xét toạ độ box, nên đề
  xuất một box khác trên trang cũ vẫn bị huỷ mà không báo lỗi rõ ràng. Phải
  luôn dặn reader tránh hẳn những trang đã dùng, không chỉ tránh trùng box.

Sau khi sửa 2 lỗi này, `vo-si-nan-nhan` vẫn chỉ đạt 3/4 vì các trang candidate
còn lại hoặc là cảnh hành động tập trung vào Ryo, hoặc dễ nhầm với một nhân
vật phụ khác (người đàn ông ria mép áo khoác đỏ mận trong mạch tống tiền "con
gái") — reader đợt 3 chủ động báo không tìm được thay vì ép ảnh sai. **Để mở,
đừng chạy thêm trừ khi có trang mới chưa từng thử.**

## Câu hỏi còn treo

- **`ogino` (荻野俊一/A Tuấn) — bí ẩn lớn nhất chưa giải quyết.** Cast ghi nhận
  anh là võ sĩ quyền anh, bị `inagaki-kouji` bắn ở tr.010, trăn trối ở tr.011,
  rồi xuất hiện tiếp ở tr.012, 015, 017–019 (mạch truyện Inagaki/Mita đấm bốc —
  có vẻ là hồi tưởng/lai lịch về Ogino, chưa xác nhận chắc). Đã `settled`.
  Nhưng ở **tr.135** (giữa mạch Makimura/chương 5), một người có ngoại hình
  khớp `ogino` xuất hiện trên sân thượng, nói câu tự giễu về "tay nghề đã rỉ
  sét" và nhắc tới một "hộ thân phù" — nếu đúng là ogino thì mâu thuẫn thẳng
  với việc anh đã chết ở tr.011. Batch đọc lúc đó tạm coi là hồi tưởng/ký ức,
  KHÔNG khẳng định. Đáng chú ý hơn: khi face-hunt cho `iwasaki` ở tr.016, phát
  hiện một bong bóng thoại nói "大家都説荻野不可能重返擂台" (mọi người đều nói
  Ogino không thể nào trở lại võ đài) — câu này nằm NGAY TRONG mạch đấm bốc
  Inagaki/Mita (tr.014–021, cùng mạch mà Iwasaki xuất hiện), gợi ý Ogino có thể
  là võ sĩ quyền anh (không chỉ nạn nhân bị bắn), và "trở lại võ đài" có thể ám
  chỉ anh còn sống hoặc mạch truyện phức tạp hơn tóm tắt hiện tại. **Cần đọc kỹ
  lại tr.007, 010–012, 014–021, và 135 liền mạch để xác định**: Ogino có thật
  sự chết không, quan hệ giữa anh với Inagaki/Mita/Iwasaki là gì, và tr.135 có
  phải hồi tưởng hay là một xuất hiện thật. Việc này ngoài phạm vi lần đọc này
  (chỉ đọc tuần tự theo batch 4 trang, không quay lại đối chiếu chéo).
- **`cung-thu-tra-thu` (cung thủ trả thù, tr.098–105) rất có thể là
  `ke-lua-gai`** (kẻ lừa gái, tr.093–095) cải trang — ngoại hình giống hệt (tóc
  đen rối, áo khoác dài be/hồng nhạt) nhưng chưa có bằng chứng lời thoại/tên để
  gộp chắc chắn. Cả hai đều `hold: true`.
- **`can-bo-yunio`** (cán bộ tổ chức 優尼奥・迪奥貝ẩn danh làm khách quen casino,
  tr.210–217) — cuối tr.216 có một hình dáng đang khóc gần giống, chưa chắc
  chắn là cùng người, để mở.
- **`chi-gai-yuko`** (đã published, 4/4 ảnh) — vẫn `hold: true`: rất có thể là
  thân chủ mới (chị gái nạn nhân yuko) được `ryo` và `nguoi-cung-cap-tin` bàn ở
  tr.044–045, cử tới quán 同伴 ở Kabukicho làm mồi nhử, nhưng chưa từng tự xưng
  tên hay được gọi tên trực tiếp trên trang — chỉ suy đoán ngữ cảnh.
- **`nguoi-cung-cap-tin`** — người cung cấp tin cho ryo, rất có thể cũng là
  người cùng anh tới Kabukicho ở các trang sau, nhưng danh tính riêng chưa rõ.
- **`nguoi-nhat-khan`** — người nhặt khăn quàng cổ của yuko đánh rơi, quan hệ
  với yuko (bạn? người thân?) chưa rõ.
- **`em-gai-con-tin`** — chưa chắc chắn có phải là "em gái" mà ryo nhắc tới
  trong lời uỷ thác đầu truyện hay không, chỉ là suy đoán ngữ cảnh.
- **`vo-si-nan-nhan` (Mita) còn thiếu 1/4 ảnh** — xem mục Face-hunt ở trên,
  đừng chạy thêm trừ khi có trang mới.

## Restore trên máy khác

```
rsync -a scan_state/cityhunter/ "/đường/dẫn/tới/City Hunter Deluxe Edition/"
```

Lưu ý thư mục thật có dấu cách ở cuối tên (`City Hunter Deluxe Edition `) —
giữ nguyên khi đường dẫn khác đi trên máy mới. Sau khi restore:

1. Sửa đường dẫn tuyệt đối trong `style_scan/claude_raw/profile_state.json`
   (liệt kê tập đã publish bằng đường dẫn tuyệt đối, giống hệt cách làm ở
   Kinnikuman — xem `scan_state/kinnikuman/RESUME.md` nếu cần script mẫu).
2. Chạy `prepare.py "<đường dẫn>" --raw-only` — phải báo **"220 already have
   notes"**. Nếu báo 0, state chưa nằm đúng chỗ, dừng lại.
3. Nếu định đọc tiếp sang **V02** (chưa có chỉ thị làm việc này — người dùng
   mới chỉ yêu cầu "Chỉ V01 trước"), cần hỏi lại trước khi mở rộng phạm vi
   đọc: `--from`/`--to` hiện chưa dùng vì lần đọc đầu chỉ trỏ thẳng vào folder
   V01 (không đọc theo `--from 1 --to N` như Kinnikuman). Sẽ cần xác nhận cấu
   trúc thư mục series trước khi chạy `prepare.py --from 1 --to 2`.
4. Cần Pillow (`pip install Pillow` trong venv riêng nếu Python hệ thống bị
   khoá externally-managed) cho `prepare.py`/`progress.py`/`finish.py`.
