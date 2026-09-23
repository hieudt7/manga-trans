# Kinnikuman — trạng thái đọc /style-read (raw-only)

Cập nhật: 2026-09-23. **Tập 1–6 đã đọc hết và publish (563/563 trang).** Việc đọc
tiếp bắt đầu từ **tập 7** trên máy khác — xem mục "Chạy tiếp trên máy khác" ngay
dưới đây trước khi làm gì khác.

| Tập | Trang | Trạng thái |
|---|---|---|
| 第01巻 | 94/94 | published |
| 第02巻 | 93/93 | published |
| 第03巻 | 94/94 | published |
| 第04巻 | 94/94 | published |
| 第05巻 | 94/94 | published |
| 第06巻 | 94/94 | published |

Cast: 127 nhân vật ghi nhận, 24 settled. Profile (đã publish): **42 nhân vật —
không còn trần 30**, 10 quy tắc xưng hô, 30 mục glossary. Face: 128 ảnh đã có,
**20 nhân vật vẫn còn thiếu ảnh** (phần lớn là nhân vật phụ đã cạn candidate
pages thật sự, không phải bỏ sót — xem "Câu hỏi còn treo").

## Đợt sửa lớn ngày 2026-09-23 — đọc trước khi đụng vào skill hay app

Ngoài việc đọc thêm 2 tập (5→6), phiên này còn sửa cả **quy tắc chọn cast** và
**cách viết personality**, cộng thêm 2 tính năng phía app. Một Claude mới cần
biết những cái này *trước khi* chạy tiếp, không thì sẽ đọc nhầm số liệu cũ hoặc
lặp lại lỗi đã sửa:

1. **Bỏ trần 30 nhân vật.** Rule chọn cast giờ là: **>10 trang HOẶC xuất hiện
   ≥2 tập** (trước chỉ có >10 trang, và cap cứng ở 30). `digest.py` giờ in thêm
   cột `volumes` cho từng nhân vật để việc này tính được bằng số, không phải áp
   dụng cảm tính. Đo thật: giao cho Gemini tự áp rule mới trên digest dài (128
   nhân vật) thì nó **bỏ sót 11 người đủ điều kiện, thêm nhầm 6 người không đủ**
   — quay lại phán đoán "ai quan trọng" thay vì đếm đúng 2 con số. Nên
   `write_profile.py` giờ tự tính sẵn danh sách "đủ điều kiện nhưng chưa có
   trong tree" bằng code rồi ép thẳng vào prompt (`MUST ADD THIS PASS` block) —
   khi tự chạy tiếp cho tập 7+, **để write_profile.py tự làm việc này, đừng tự
   tay soát danh sách qua digest**.
2. **`write_profile.py` giờ gọi Gemini 2 lượt, không phải 1.** Lượt 1 chỉ viết
   cây quan hệ (relations/address), để trống `role`/`personality`/`speech`.
   Lượt 2 mới viết 3 trường đó, dựa trên cây quan hệ lượt 1 vừa chốt làm ngữ
   cảnh. Đo thật: gộp chung 1 lượt thì personality viết hời hợt hơn hẳn — model
   dồn hết chú ý vào phần cơ học (ai gọi ai) và personality bị lơ là. Không cần
   làm gì thêm để dùng cái này — `write_profile.py` tự động chạy 2 lượt, chỉ
   cần biết là 1 lần "Fold volume X" giờ tốn 2 lần gọi Gemini, không phải 1.
3. **`Meat` từng bị ghi nhầm thành "Mít"** ở 9 chỗ rải rác (personality/role/
   relations của các nhân vật KHÁC nhắc tới meat bằng tên) dù field `name` của
   chính meat đã đúng — vì lượt viết profile trước chỉ sửa tên nhân vật, không
   quét hết toàn bộ text tham chiếu tới tên đó ở người khác. Đã sửa hết, đã
   republish. Nếu thấy tên lạ tương tự lặp lại ở nhân vật mới (tên chế thay vì
   phiên âm katakana), quét theo cách này: sửa cả field riêng lẫn mọi chỗ nhắc
   tên đó trong `role`/`personality`/`speech`/`relations[].address` của
   NGƯỜI KHÁC.
4. **`western-girl` hoá ra chính là `natsuko`** (mọi trang đều ghi rõ "ナツコ"
   trong tranh, chỉ khác là đang mặc đồ cao bồi cho 1 chương riêng) — đã gộp
   pages vào natsuko, xoá entry `western-girl`. Nếu gặp case tương tự (2 id
   cho cùng 1 người, khác trang phục/chương), **tự mình xem ảnh xác nhận trước
   khi gộp** — đừng để reader/subagent tự gộp.
5. **App giờ có tính năng đọc "ai nói dòng thoại nào, với ai" bằng Gemini
   vision** (`speaker_attribution.rs`), thay vì chỉ đoán bằng khoảng cách hình
   học tới khuôn mặt gần nhất (cách cũ mù hoàn toàn với nhân vật đeo mặt nạ —
   tức gần cả dàn Kinnikuman). Để tính năng này thật sự chạy khi dịch:
   - **Style Scanner** → chọn đúng profile Kinnikuman → bấm **"Use for
     translation"** (bắt buộc lại mỗi lần profile được republish — nó copy
     sang 1 file snapshot riêng, không tự đồng bộ).
   - **Character Scanner** → bấm **"Sync All to Library"** (nút mới) để CV
     face-scan nhận diện đúng nhân vật đang có mặt trên trang.
   - Trang **Folder** có switch **"Use character context"** (mặc định bật) —
     giữ nguyên, đừng tắt trừ khi cố tình test không có ngữ cảnh.
   - Máy mới cần **build lại app** (`cargo build`/`bun run dev` từ gốc repo)
     sau khi pull code mới — tính năng này nằm ở backend Rust, không tự có
     nếu chỉ restore state theo mục dưới.

## Chạy tiếp trên máy khác

0. **Pull code mới trước tiên** (`git pull` trong repo `manga-trans`). State ở
   `scan_state/` và code skill/app đi cùng nhau trong cùng mấy commit ngày
   2026-09-23 — restore state mà chạy code cũ (skill cũ vẫn còn trần 30, app
   cũ chưa có speaker-attribution) sẽ cho kết quả khác với những gì mô tả ở
   trên.

1. **Khôi phục state vào thư mục truyện** (giữ nguyên tên thư mục tập):

   ```
   rsync -a scan_state/kinnikuman/ "/đường/dẫn/tới/Kinnikuman/"
   ```

2. **Sửa đường dẫn tuyệt đối trong `profile_state.json`.** File này liệt kê các tập đã
   publish (giờ là **6 tập**) bằng **đường dẫn tuyệt đối**, và `progress.py` so khớp đúng
   chuỗi đó. Nếu username hoặc vị trí thư mục khác đi, cả 6 tập sẽ bị báo nhầm là *"read
   but not yet in the profile"* và bị viết lại profile vô ích. Sửa trước khi chạy:

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
   <venv>/bin/python .claude/skills/style-read/prepare.py "/đường/dẫn/tới/Kinnikuman" --from 1 --to 6 --raw-only
   ```

   Phải báo **"563 already have notes"**. Nếu báo 0 → state chưa nằm đúng chỗ, **dừng lại**,
   đừng đọc lại từ đầu.

   Rồi mở rộng ra tập 7 (và bao nhiêu tập định đọc tiếp), việc này sẽ ghi thêm ảnh/manifest
   cho các trang mới mà không đụng gì tới 563 trang đã có note:

   ```
   <venv>/bin/python .claude/skills/style-read/prepare.py "/đường/dẫn/tới/Kinnikuman" --from 1 --to <tập cuối định đọc> --raw-only
   <venv>/bin/python .claude/skills/style-read/face_hints.py "/đường/dẫn/tới/Kinnikuman" --raw-only
   ```

   Bước `face_hints.py` không bắt buộc nhưng nên chạy — cho mỗi trang mới một face box
   tính sẵn bằng CV, đỡ phải áng chừng bằng mắt lúc đọc và lúc face-hunt.

5. **Đọc tiếp từ tập 7, trang đầu tiên.** Cách đọc dưới đây không đổi so với lúc đọc
   tập 5–6, chỉ có FACES_PER_CHARACTER đã lên 4 (xem `.claude/agents/manga-page-reader.md`):

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

   Lần chạy `prepare.py` đầu tiên trên máy mới sẽ sinh thêm ảnh nửa trang cho các
   trang đã có note và render lại reading copy. Chỉ tốn CPU, không tốn token, và
   không đụng vào note đã có.

   Hết mỗi tập thì chạy `write_profile.py --volume <N>` — **đừng tự viết
   `profile.json` bằng tay, và đừng tự đọc `digest.py --cast` rồi suy ra ai đủ
   điều kiện**: script tự tính danh sách "đủ điều kiện nhưng chưa có trong tree"
   (mục "Đợt sửa lớn" ở trên) và tự chạy đúng 2 lượt Gemini (tree rồi personality).
   Việc của người chạy chỉ là đọc summary nó in ra, kiểm `verify_profile.py`, rồi
   chạy `finish.py`.

   **Không bao giờ xoá `style_scan/claude_raw/` để "chạy lại cho sạch"** — mất hết
   563 trang đã đọc. `updates/applied/` cũng đừng xoá: đó là nhật ký từng lượt.

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
  Cập nhật 2026-09-23: `nakano-kazuo` giờ đã publish (đủ điều kiện rule mới, xuất hiện ≥2 tập).
  Face-hunt phát hiện thêm: trang v03-0064 có nhân vật ghi rõ tên "和雄" nhưng vẽ khác hẳn (đầu trọc,
  đeo kính, không râu — so với ông già tóc xù bạc trắng ở v02-0006) — **không gộp**, giữ nguyên là
  face của v02-0006 thôi, nhưng đây có thể chính là bằng chứng cho câu hỏi "1 người hay 2" ở trên.
  Cần ai đó đọc kỹ trang v03-0064 để xác định.
- **terry-girl** (tóc dài gợn sóng, thân với Terryman) — vẫn là entry riêng, chưa có tên. (`western-girl`
  đã xác định KHÔNG phải trường hợp tương tự — xem mục "Đã giải quyết".)
- **kamehame** nói *"これで わたしの役目はおわった…"* sau khi nhận lại đai, nhưng chưa thấy cảnh ông đi.
- `robin-mask` và `terryman` vẫn thiếu `ageGroup` (mặt nạ không bao giờ tháo hẳn).
- Vài câu chưa gán được người nói ở tập 5: *"みそこなったよキン肉マン…"*, *"うせろ/負け犬"*, "カスタ" là ai,
  người tóc xoăn áo sọc ở Jesse Palace, người hộ tống đeo kính râm ở LA, đô vật da đá bắt tay
  Kinnikuman ở v05-040.
- **`mayumi` vẫn thiếu ảnh mặt** (mới có 1/4, cần 3) sau **2 đợt face-hunt liên tiếp đều fail**: mọi
  candidate page đều chỉ cho thấy mặt nạ kín có dấu "王" giống hệt `king-muscle`, không lần nào thấy
  râu (mô tả mayumi là "có râu"). Đợt 2 subagent đã đúng đắn báo 0 thay vì đoán liều lần 3 — **đừng
  chạy thêm face-hunt cho mayumi trừ khi có trang mới chưa từng thử**; để mở, đừng ép ảnh sai vào lần
  nữa. Không loại trừ khả năng mayumi thật ra không có cảnh nào lộ mặt trần trong cả bộ truyện.
- **13 câu trích dẫn trong profile không tìm thấy trong notes** (`verify_profile.py` báo, chưa xử lý):
  `natsuko` (speech + →kinnikuman), `robin-mask` (→terryman), `ramenman` (speech + →kinnikuman),
  `kamehame` (→kinnikuman), `beauty-rhodes` (speech + →doro-flairs), `sheik-seijin` (speech +
  →kinnikuman), `iyadesu-harisun` (→kinnikuman ×2). Đây là câu bị chép sai trang, viết lại khác ý gốc,
  hoặc bịa — `finish.py` không bắt được (chỉ kiểm hình dạng), phải mở note gốc bằng `digest.py --cast`
  rồi grep tìm câu thật, hoặc bỏ claim nếu không tìm thấy. Chưa ai xử lý — cần làm trước khi coi
  profile là "sạch".
- Tên "Meat" cho ミート đã sửa lại đúng theo phiên âm katakana (đợt 2026-09-22, rồi phát hiện lại lần
  2 ở 2026-09-23: field `name` đã đúng nhưng "Mít" vẫn còn kẹt trong `personality`/`role`/`relations`
  của 9 nhân vật KHÁC nhắc tới meat bằng tên — xem mục "Đợt sửa lớn" ở đầu file để biết cách quét cho
  hết lần sau).

## Đã giải quyết trong đợt 2026-09-23

- **Đọc xong tập 5 và 6** (563/563 trang), publish cả 6 tập.
- **`western-girl` chính là `natsuko`** — mọi trang được cho là "western-girl" đều ghi rõ "ナツコ"
  trong tranh, chỉ là đang mặc đồ cao bồi cho chương「よみがえる西部魂の巻」. Đã gộp pages vào
  `natsuko`, xoá entry `western-girl`. (`terry-girl` thì KHÔNG cùng trường hợp — vẫn treo, xem trên.)
- **Cast rule đổi, cast tăng 29 → 42**: thêm `yosaku, queen, brockenman, sky-man, panda-man,
  western-girl(→natsuko), palace-maiden, medium-rare, akaiwa, nakano-kazuo, alisa, chabo-kerori,
  stone-schemer` — toàn bộ xuất hiện ≥2 tập dù mỗi tập chỉ vài trang.
- **Face: 31 → 128 ảnh**, rà tay từng ảnh một qua 3 đợt, loại 21 ảnh sai (không phải hàng loạt máy
  ép bừa — mỗi ảnh đều bị soi riêng, không ghép sheet).
- **Tên "Mít" quét sạch khỏi mọi chỗ** (field riêng + 9 chỗ nhắc tên ở nhân vật khác + glossary).

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
