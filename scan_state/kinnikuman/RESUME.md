# Kinnikuman — trạng thái đọc /style-read (raw-only)

Cập nhật: 2026-09-20 (skill đã tối ưu token — xem mục 5). Đọc tới **436/469 trang**. Tập 1–4 đã publish, tập 5 đang dở ở **61/94**.

| Tập | Trang | Trạng thái |
|---|---|---|
| 第01巻 | 94/94 | published |
| 第02巻 | 93/93 | published |
| 第03巻 | 94/94 | published |
| 第04巻 | 94/94 | published |
| 第05巻 | 61/94 | đang đọc — trang kế: `v05-061` |

Đo thật trên v05-061..064 (xem mục 5): input **577.040 → 66.206 token/trang**,
số lần gọi API **5,8 → 1,25 mỗi trang**. Nhưng output (suy luận) tăng 8.144 → 14.292,
nên quy ra tiền API chỉ giảm nhẹ — output giờ mới là khoản đắt nhất.

Cast: 115 nhân vật, 22 settled. Profile: 30 nhân vật (mức trần), 14 quy tắc xưng hô, 60 mục glossary.

## Chạy tiếp trên máy khác

1. **Khôi phục state vào thư mục truyện** (giữ nguyên tên thư mục tập):

   ```
   rsync -a scan_state/kinnikuman/ "/đường/dẫn/tới/Kinnikuman/"
   ```

2. **Sửa đường dẫn tuyệt đối trong `profile_state.json`.** File này liệt kê các tập đã
   publish bằng **đường dẫn tuyệt đối**, và `progress.py` so khớp đúng chuỗi đó. Nếu
   username hoặc vị trí thư mục khác đi, tập 1–4 sẽ bị báo nhầm là *"read but not yet in
   the profile"* và bị viết lại profile vô ích. Sửa trước khi chạy:

   ```
   python3 - <<'EOF'
   import json
   p = "/đường/dẫn/tới/Kinnikuman/style_scan/claude_raw/profile_state.json"
   NEW = "/đường/dẫn/tới/Kinnikuman"
   s = json.load(open(p))
   s["published"] = [NEW + x[x.rindex("/["):] for x in s["published"]]
   json.dump(s, open(p, "w"), ensure_ascii=False, indent=2)
   print(s)
   EOF
   ```

3. **Cần Pillow.** `prepare.py` / `progress.py` / `finish.py` đều `import PIL`
   (`digest.py` thì không).
   Homebrew Python là externally-managed nên không `pip install` thẳng được; tạo venv riêng:

   ```
   python3 -m venv /tmp/styleread-venv && /tmp/styleread-venv/bin/pip install Pillow
   ```

   rồi gọi các script bằng `/tmp/styleread-venv/bin/python` thay cho `python3`.

4. **Chạy prepare rồi kiểm tra con số.**

   ```
   <venv>/bin/python .claude/skills/style-read/prepare.py "/đường/dẫn/tới/Kinnikuman" --from 1 --to 5 --raw-only
   ```

   Phải báo **"436 already have notes"**. Nếu báo 0 → state chưa nằm đúng chỗ, **dừng lại**,
   đừng đọc lại từ đầu.

5. **Đọc tiếp từ `v05-061`.** Skill đã đổi từ 2026-09-20, chạy theo cách mới:

   - mỗi lượt **4 trang** (trước là 6), tuần tự từng lượt — không song song.
     Chi phí một lượt = số lần gọi tool × context mỗi lần, mà ảnh và note của
     trang trước nằm lại trong context đến hết lượt, nên lượt càng dài càng đắt
     theo cấp số nhân, không phải tuyến tính;
   - reader là subagent kiểu **`manga-page-reader`** (`.claude/agents/`), đã mang
     sẵn toàn bộ brief; prompt chỉ gồm danh sách trang + đường dẫn;
   - reader **không mở `cast.json`** nữa. Nó đọc `known.md` và ghi phát hiện vào
     `style_scan/claude_raw/updates/<page-id>.json`; `progress.py` gộp vào cast
     rồi dời sang `updates/applied/`;
   - reader chỉ có `Read` + `Write`, **không có Bash nên không crop được**. Bù lại
     `prepare.py` cắt sẵn và phóng to nửa trang lên 993x1568 (so với 760px/trang
     nếu đọc cả spread);
   - **mỗi trang chỉ ghi MỘT file**: note, kết thúc bằng khối `### Cast` chứa JSON.
     `progress.py` bóc khối đó ra, gộp vào cast, ghi nhận vào `cast_applied.json`
     nên không gộp lại lần hai;
   - **reader chỉ đọc hai NỬA trang** (phải trước, trái sau), không đọc spread —
     trước đây nó mở spread rồi lại mở nửa của đúng trang đó, tức trả tiền hai lần.
     Spread chỉ dùng khi tranh vắt ngang gáy. Face box ghi kèm `"half": "right"|"left"`,
     `finish.py` quy đổi về toạ độ spread;
   - sau mỗi lượt vẫn chạy `progress.py`.

   Lần chạy `prepare.py` đầu tiên sẽ sinh thêm ảnh nửa trang cho cả 469 trang và
   render lại reading copy ở 1568px. Chỉ tốn CPU, không tốn token, và không đụng
   vào note đã có.

   Hết tập 5 thì viết `profile.json` từ `digest.py --volume 5` và `digest.py --cast`
   (đừng đọc từng note hay đọc thẳng `cast.json` — cast đã 7.500 dòng, vượt xa
   một lần đọc), rồi chạy `finish.py`.

   **Không bao giờ xoá `style_scan/claude_raw/` để "chạy lại cho sạch"** — mất hết
   436 trang đã đọc. `updates/applied/` cũng đừng xoá: đó là nhật ký từng lượt.

## Lưu ý khi giao việc cho reader

Ba điều dưới đây **đã được đưa hẳn vào `.claude/agents/manga-page-reader.md`**,
không cần nhắc lại trong prompt mỗi lượt nữa — ghi ở đây để biết vì sao chúng có
trong brief:

- **Thứ tự trang:** mỗi file scan là một spread hai trang, đọc **nửa PHẢI trước,
  nửa TRÁI sau**, panel trong mỗi nửa cũng phải sang trái, và đối chiếu số trang
  ở chân trang. Một reader ở tập 3 đã đoán ngược lúc đầu. Vài spread ở tập 1–2 có
  thể còn bị đảo thứ tự nội bộ.
- **Cameo:** nhân vật gag/đám đông một lần không vào cast — riêng arc Olympic và
  các trang bìa chương có rất nhiều đô vật do độc giả gửi. Chỉ ai thật sự đánh một
  trận, được gọi tên, hoặc thoại ở hơn một trang mới cho vào.
- **Không tự gộp id đang `hold`** — báo về để người chạy quyết định.

Còn lại một việc phải tự tay làm: **`king-muscle` đã settled** nên reader được
dặn không xem lại. Nếu cần sửa mục `トンカツ屋のイクヱちゃん` nằm nhầm dưới id này
(xem phần dưới), phải sửa trực tiếp `cast.json` — update file không ghi đè được
trường đã có.

## Câu hỏi còn treo

Đang `hold: true`: `pig-impostor`, `king-tone`, `nakano-kazuo`, `kazuo`, `stone-schemer`, `eyestalk-schemer`.

- **vengeful-stranger** — ông chú sẹo mặt, áo caro. Cuối tập 3 nói với Natsuko *"おれはキン肉マンを
  殺す……!"* rồi túm lấy cô; suốt tập 4 lại thành người kèm góc đài cho Kinnikuman, mách chiến thuật
  hạ Robin Mask, cứu Robin bằng súng rã đông, an ủi anh sau trận. Ai cũng gọi là おじさん.
  **Tên thật và quan hệ với Natsuko vẫn chưa được giải thích** — đây là đầu mối lớn nhất còn treo.
- **king-tone / pig-impostor** — hai nhân vật mặt lợn, có thể là một. Chưa có trang nào xác nhận.
  Đã thấy vài nhân vật mặt lợn khác (ukon, một bóng người đeo dấu "K" ở tập 4) nhưng **không** liên quan.
- **トンカツ屋のイクヱちゃん** (hồi tưởng v01-0321) hiện ghi dưới `king-muscle`, nhiều khả năng là quá
  khứ của `king-tone`. `king-muscle` đã settled nên reader không xem lại, và update file cũng
  không ghi đè được trường đã có — muốn sửa thì sửa thẳng `cast.json`.
- **nakano-kazuo (中野和雄)** và **kazuo (和雄)** trùng tên và trùng gag tóc giả "アデランスの和雄".
  Chưa rõ là một người hay hai. Cả hai **khác** `kazu-nakano` (カズ・ナカーノ, hướng dẫn viên Hawaii).
- **western-girl** (mũ lưỡi trai, yếm bò, gọi テリー) và **terry-girl** (tóc dài gợn sóng) — hai entry
  riêng, đều thân với Terryman, đều chưa có tên.
- **kamehame** nói *"これで わたしの役目はおわった…"* sau khi nhận lại đai, nhưng chưa thấy cảnh ông đi.
- `robin-mask` và `terryman` vẫn thiếu `ageGroup` (mặt nạ không bao giờ tháo hẳn).
- Vài câu chưa gán được người nói ở tập 5: *"みそこなったよキン肉マン…"*, *"うせろ/負け犬"*, "カスタ" là ai,
  người tóc xoăn áo sọc ở Jesse Palace, người hộ tống đeo kính râm ở LA, đô vật da đá bắt tay
  Kinnikuman ở v05-040.
- Tên "Meat" cho ミート đã sửa lại đúng theo phiên âm katakana (2026-09-22, trước đó bị viết nhầm thành "Mít" — một tên tự chế không liên quan đến ミート).

## Đã giải quyết trong đợt này

- **`skull-schemer` chính là `kinkotsuman`** — đã gộp id. Chuỗi bằng chứng: Kinnikuman gọi mỉa
  "キン骨マンちゃん" (v03-0020), bài hát chế "キン骨マ〜ン" (v03-0030), đàn em gọi 先生 giống tay sai
  của Kinkotsuman, hắn gọi bộ đàm cho イワオ (tay sai riêng của Kinkotsuman), và cuối cùng bị gọi
  thẳng "キン骨マン!!" giữa cảnh đối đầu nghiêm túc (v04-011). Hai id chưa từng xuất hiện chung một
  trang. Đã gộp faces + addresses, viết lại 16 note, xoá entry cũ.
- **ビッグマウンテン** chỉ là biệt danh của `jesse-mayvia`, không phải nhân vật mới (v05-020).
- **シャネルマン** là `kinnikuman` cải trang, không phải nhân vật riêng (v05-057) — một reader đã
  tạo nhầm entry rồi tự rút lại.
- Biệt danh **キンちゃん** xác nhận là của `natsuko` gọi Kinnikuman (v03-0077, v04).
- `karekkuku` đã có giọng (v03-0057: giọng trẻ con hoảng sợ). `harigorasu`, `daiking`, `tendon`,
  `manmora`, `nagaashi-gon` đều đã lộ diện đầy đủ.
- `village-hag` **không** phải vợ `nakano-kazuo` (vợ ông tên 公子/Kimiko, là người khác).

## Phát hiện đáng chú ý cho người dịch

- **Đại từ kiểu Tây hoá ミー/ユー là dấu hiệu "người nước ngoài" chung cho cả dàn**, không riêng
  Terryman: Robin Mask (Anh), Gania Mask (huấn luyện viên của Robin), Kazu Nakano (Việt kiều kiểu
  Nisei), Duke Kamata đều dùng. Nên chọn **một** cách dịch và dùng nhất quán cho tất cả.
- **Giận không đồng nghĩa với mất kính ngữ.** Câu lạnh nhất của Robin Mask — tuyên bố sẽ giết
  Kinnikuman — vẫn giữ thể ます: 「死んでもらいます!」. Meat cũng giữ です/ます ngay cả khi trách
  hoàng tử. Đừng san phẳng mọi câu giận thành "tao/mày".
- **Quái vật trong tuần hiếm khi là kẻ ác**: Nachiguron, shell-monster, Nagaashi Gon, Tendon,
  frog-monster đều là những sinh vật cô đơn/kiêu hãnh/tuyệt vọng, chương của họ kết bằng thương
  cảm hoặc hoà giải. Giữ giọng Việt có cảm thông, đừng biến thành giọng ác nhân chung chung.
- Brockenman mang tạo hình phát xít và đòn kết liễu được vẽ như phòng hơi ngạt — đúng nguyên tác
  1979. Dịch đúng những gì có trên trang và báo biên tập, đừng tự ý làm nhẹ đi.
