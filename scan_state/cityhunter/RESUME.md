# City Hunter — trạng thái đọc /style-read (raw-only)

Cập nhật: 2026-09-28. **Tập 2 (Deluxe Edition V02) đã đọc hết và publish
(216/216 trang)**, đọc theo series (`--from 2 --to 2`) trên cùng cast với tập
1 — series root là thư mục có dấu cách ở cuối: `City Hunter Deluxe Edition `
(work dir `style_scan/claude_raw/` nằm ngay dưới root này, không phải dưới
từng volume). Vẫn chưa có bản dịch tiếng Việt — mọi tên riêng/xưng hô là gợi ý
của reader.

## Việc lớn nhất đợt này: gộp thủ công 3 id trùng thành 1 nhân vật (Sayaka)

Ba batch khác nhau (không thấy ngữ cảnh của nhau) đặt **3 id riêng** cho cùng
một nhân vật — con gái `ryujin-nobuo` (thân chủ mới) — vì cô xuất hiện ở 3 bối
cảnh trang phục khác hẳn nhau trải dài ~90 trang:
`con-gai-ryujin-nobuo` (đồng phục học sinh, tr.122–137, mở đầu mạch),
`hoc-tro-gia-su` (áo ngủ ở nhà khi Ryo cải trang gia sư, tr.139–170),
`nu-tu-nhan-xich` (bị băng Buruo Isutan bắt cóc xích lại, tr.185–203). Mỗi
batch đều làm đúng quy trình — chỉ báo nghi vấn trùng tên "沙也加" trong note,
không tự gộp. Đã xác nhận gộp bằng cách tự crop và nhìn trực tiếp ảnh mặt ở cả
3 id (không chỉ tin lời note): 2/4 face box của `con-gai-ryujin-nobuo` hoá ra
là ảnh RYO (kính tròn) bị gán nhầm, 1 box khác là mảnh kính vỡ — chỉ 1/4 box
là mặt thật; sau khi loại các box sai, mặt còn lại khớp rõ với
`hoc-tro-gia-su` và `nu-tu-nhan-xich`. Bằng chứng cốt truyện: cha là
"老狐狸"/`ryujin-nobuo`, tổ chức hôn nhân sắp đặt `青堅會`/`龍神會` nhắc lại ở cả
3 mạch, và thứ tự chương liền mạch (thân chủ → gia sư → bị bắt cóc).

**Đã gộp thành `con-gai-ryujin-nobuo`** (giữ id này, KHÔNG dùng "sayaka" —
xem bẫy kỹ thuật bên dưới). Một id thứ 4 trùng tên, `sayaka-bi-bam-duoi`
(tr.175–176, một cô gái tên Sayaka khác bị cha hói đầu béo theo dõi bằng
thang treo vệ sinh toà nhà), **KHÔNG gộp** — ngoại hình/bối cảnh cha hoàn toàn
khác, chỉ là trùng tên phổ biến (giống hệt kiểu bẫy "trùng tên" đã gặp ở tập
1). Để mở.

**Bẫy kỹ thuật khi chọn id gộp:** thử đặt id gộp là `"sayaka"` trước — vỡ
ngay, vì `names_to_ids()` trong `progress.py` build lookup từ CẢ id lẫn
trường `name` của mọi nhân vật, và `sayaka-bi-bam-duoi` cũng có
`"name": "Sayaka"`. `setdefault()` khiến `sayaka-bi-bam-duoi` (đứng trước
trong list) chiếm mất khoá `"sayaka"`, nên mọi note ghi "Nhân vật: sayaka"
bị `who()` map nhầm sang `sayaka-bi-bam-duoi` — `evidence()` trả về 0 trang
cho nhân vật vừa gộp dù đã đổi tên đúng trong toàn bộ note. Càng nguy hiểm
hơn vì id ngắn `sayaka` là **substring** của `sayaka-bi-bam-duoi`, nên lệnh
`sed s/sayaka/.../g` đầu tiên (đổi 3 id cũ → "sayaka") khi bị đổi ngược lại
cũng vô tình phá luôn `sayaka-bi-bam-duoi` thành
`con-gai-ryujin-nobuo-bi-bam-duoi` — phải rà lại bằng `grep` cả hai chiều và
sửa tay. **Rút kinh nghiệm cho lần sau:** không bao giờ chọn id gộp trùng với
trường `name` (hiển thị) của một nhân vật khác đang mở, kể cả khi id đó nghe
"sạch" hơn; và khi rename id bằng `sed` trên toàn bộ notes, luôn kiểm tra id
đó có phải substring của id nào khác không trước khi chạy.

**Hai nghi vấn khác đã tự tay xem ảnh và KHÔNG đủ căn cứ để gộp** (để mở,
đừng tự gộp nếu không có thêm bằng chứng mới):
- `tu-nhan-mu-canh-sat` (tù nhân đeo mũ cảnh sát, xích chung với
  `con-gai-ryujin-nobuo`, tr.185+) — nhiều batch nghi là Ryo cải trang (nhắc
  "gia sư", gài máy nghe lén, đùa cợt lém lỉnh), Gemini ở bước viết profile
  cũng tự suy luận vậy trong `role`. Nhưng ảnh mặt duy nhất có được không đeo
  kính tròn đặc trưng của Ryo — không đủ chắc để gộp vào `ryo`.
- `nguoi-choang-trang-bi-an` (tr.063–068, kẻ mặc đồ trắng lao qua kính vỡ,
  giết 26 người của Long Thần Hội, luôn hét tên "Saeba") — mô tả ban đầu
  ("tóc đen chải ngược, áo choàng trắng") trùng khớp `tuong-quan`, nhưng khi
  tự crop ảnh mặt ở tr.064 thì hoá ra là một người tóc xoăn cầm mic hát
  karaoke, hoàn toàn khác `tuong-quan` — rất có thể chỉ là khách qua đường,
  KHÔNG liên quan. Vẫn để nguyên là id riêng, chưa rõ tên.

**Lưu ý quan trọng về nguồn "raw":** bản scan Deluxe Edition này đã bị rebubble
lời thoại sang **tiếng Trung phồn thể**, không phải tiếng Nhật gốc. Theo đúng
yêu cầu ban đầu, quy trình đọc coi hẳn tiếng Trung này là "raw" — đọc, trích
dẫn, phiên âm y như đang đọc tiếng Nhật, không dịch nó ra tiếng Việt trong ghi
chú. Chữ Nhật gốc (biển hiệu, SFX, ngày tháng Chiêu Hoà) vẫn còn giữ nguyên ở
hậu cảnh và không bị đụng tới.

| Tập | Trang | Trạng thái |
|---|---|---|
| Deluxe Edition V01 | 220/220 | published |
| Deluxe Edition V02 | 216/216 | published |

Cast tích luỹ (cả 2 tập, 1 cast.json chung): 72 nhân vật ghi nhận, 6 settled
(`ogino, ryo, makimura, ryujin-nobuo, trum-buruo-isutan, con-gai-ryujin-nobuo`).
Profile V01 (7 nhân vật, xem đợt cập nhật 2026-09-24 bên dưới) + V02 thêm
**20 nhân vật mới** → **27 nhân vật** trong profile sau khi publish V02.

Face V02: **21 ảnh** sau 1 vòng face-hunt + soát tay (loại 8 box sai — xem
"Soát ảnh mặt V02" bên dưới). Vẫn còn **17 nhân vật** V02 thiếu ảnh đủ 4/4,
phần lớn là nhân vật phụ chỉ có 1 trang candidate đã thử và xác nhận không có
mặt rõ (xem `face_gaps.json`) — **để mở, đừng chạy lại cùng trang cũ**:
`ke-bi-tiem-angel-dust, dan-em-buruo-isutan, truong-lao (không bao giờ lộ
mặt), nhan-vien-general, nguoi-cung-cap-tin, ke-uy-hiep-makimura,
gai-bay-my-nhan, bao-ve-toc-hoi, nguoi-say-ruou, dan-em-lo-lang,
trum-vua-ra-tu, tinh-dich-makimura, nu-trum-yunio, nguoi-thieu-no` (đều còn
0–1/4), cộng `tuong-quan, ryujin-nobuo, con-gai-ryujin-nobuo, trum-buruo-isutan,
ga-bavaro, tu-nhan-mu-canh-sat` (đã có 1–3/4, thiếu vài ảnh để đủ 4/4).

## Soát ảnh mặt V02 — bẫy: subagent viết nhầm `.md` thay vì `.json`

Đợt face-hunt đầu tiên viết kết quả vào
`updates/facehunt-v02.md` (đúng format khối ```json` như note, nhưng sai định
dạng file) — `apply_updates()` trong `progress.py` chỉ quét file `.json`
trong thư mục `updates/`, **không đọc `.md`**, nên `progress.py` chạy xong báo
"folded in" nhưng số ảnh không tăng, không có lỗi rõ ràng nào cả. Phải tự phát
hiện qua việc so `finish.py` in ra cùng một số "vẫn thiếu N nhân vật" hai lần
liên tiếp. Sửa: trích khối JSON ra, lưu thành `.json` trực tiếp trong
`updates/`. **Nhớ cho lần sau: dặn subagent face-hunt ghi thẳng file
`.json`, không phải `.md` có nhúng khối JSON** (khác với note của một batch
đọc thường, vốn LÀ `.md`).

Sau khi fold đúng, soát bằng mắt toàn bộ 29 ảnh trong `character_scan/faces/`
(dùng 1 subagent chỉ có quyền Read, không xem note/thoại) phát hiện **8/29
ảnh sai** — chủ yếu là bong bóng thoại bị crop nhầm thành "mặt", một crop lẫn
2 khuôn mặt trong 1 khung, và 2 trường hợp gán nhầm sang nhân vật khác hẳn
(`ryujin-nobuo/3` hoá ra là mặt con gái ông ta; hai box trong
`tu-nhan-mu-canh-sat` từ vòng face-hunt đầu — trước khi soát — hoá ra là một
người đàn ông có ria mép hoàn toàn khác, đã bị loại ngay khi tự crop kiểm tra
trước khi fold, không đợi tới bước soát ảnh). Đã xoá cả 8 bằng
`review_faces.py --bad id/index,...` (không sửa `cast.json` tay) rồi chạy lại
`finish.py`. **Kết luận: dù reader tự nói "đã crop-kiểm tra" trong note, vẫn
phải tự mở ảnh crop cuối cùng mà xem — pattern lặp lại nhiều lần trong đợt
này là gán nhầm mặt sang đúng loại nhân vật (nam có ria mép, mũ lưỡi trai)
nhưng SAI người cụ thể.**

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
2. Chạy `prepare.py "<đường dẫn tới series root, giữ dấu cách cuối tên>" --from 2 --to 2 --raw-only`
   — phải báo **"216 already have notes"** cho V02 (và tương tự cho V01 nếu
   chạy `--from 1 --to 1`). Nếu báo 0, state chưa nằm đúng chỗ, dừng lại.
   **Đính chính so với ghi chú cũ ở đây:** V01 thực ra ĐÃ được đọc theo series
   mode (`--from`/`--to`) ngay từ đầu, không phải trỏ thẳng vào folder con như
   từng ghi nhầm — bằng chứng: work dir `style_scan/claude_raw/` nằm ở series
   root (`City Hunter Deluxe Edition /`), không phải trong
   `.../V01/style_scan/`, và mọi page id đều có tiền tố `v01-`. V02 (đọc
   2026-09-28) tiếp tục đúng convention này với `--from 2 --to 2`, dùng chung
   1 cast/profile với V01. Đọc tập tiếp theo (V03) cũng theo mẫu này:
   `--from 3 --to 3`.
3. Cần Pillow (`pip install Pillow` trong venv riêng nếu Python hệ thống bị
   khoá externally-managed) cho `prepare.py`/`progress.py`/`finish.py`.
