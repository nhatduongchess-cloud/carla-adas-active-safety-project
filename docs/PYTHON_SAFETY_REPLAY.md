# Xác minh replay AEB trong CARLA (Python)

`ActiveSafetySystem.update()` chuẩn bị đầu vào chưa làm tròn, sau đó gọi
`SafetyFSM.step()`. Phần hình học vẫn giữ lịch sử khoảng cách LiDAR.
Máy trạng thái giữ riêng bộ đếm né và chốt phanh theo episode. Replay ghi giá
trị ma sát và hệ số khoảng cách đang có hiệu lực trước lần gọi FSM. Trong
mô phỏng, `EgoController` tiếp tục sở hữu lệnh điều khiển xe.

## Baseline và kiểm thử hồi quy

Baseline được ghi từ mã Python gốc tại commit
`fa83772c6c14c7a78d10bdd8684ce6e5f6a974be` trước khi tách FSM.
[Fixture gốc](../tests/fixtures/safety/python_baseline.json) chứa hash của mã
tham chiếu, cấu hình, 29 bước trong 7 episode tổng hợp và các ca phân xử
AEB/MRM. Môi trường baseline chạy 210 self-check và 310 unit test theo
danh sách offline trong workflow trên máy lúc thử. Fixture đối chiếu đầy đủ
`SafetyDecision` và lý do chuyển trạng thái.

Chạy kiểm thử và fixture replay từ thư mục gốc bằng Python của dự án.

```powershell
python selftest.py
python -m unittest tests.test_safety_replay
python scripts/tools/compare_safety_replay.py --input tests/fixtures/safety/python_replay --report logs/safety-fixture-comparison.json
```

## Ghi một lượt CARLA

Với CARLA đã sẵn sàng, chạy ca HardBrake trên Town02. Dùng tên thư mục mới
cho mỗi lượt capture. Kịch bản 20 giây ở 40 Hz có 800 bước chính và 59 bước
warm-up đã gọi FSM.

```powershell
python run_scenarios.py --scenarios HardBrake --town Town02 --seed 42 --weather clear --seconds 20 --inference-device cpu --safety-replay-out logs/hard-brake-capture --report logs/hard-brake-scenario.json
python scripts/tools/compare_safety_replay.py --input logs/hard-brake-capture/case-001 --report logs/hard-brake-comparison.json
```

Ứng dụng chính cũng nhận `--safety-replay-out` và đếm mọi lần gọi AEB,
kể cả các quyết định `DRIVE`.

```powershell
python chinh.py --vehicles 0 --town Town02 --seed 42 --no-display --duration 5 --inference-device cpu --safety-replay-out logs/runtime-aeb-capture --run-report logs/runtime-aeb.json
python scripts/tools/compare_safety_replay.py --input logs/runtime-aeb-capture --report logs/runtime-aeb-comparison.json
```

Mỗi capture có `manifest.json` và `steps.jsonl`. Manifest chứa cấu hình hiệu
lực, hash cấu hình, bối cảnh chạy và tổng số bước. Mỗi dòng có sequence ID
riêng của episode, world frame nếu có, trạng thái cảm biến, quyết định Python
và dữ liệu phân xử. `null` trong khoảng cách/TTC biểu thị thiếu kết quả hữu
hạn. Writer ghi theo luồng với bộ đệm file hữu hạn và chốt manifest khi kết
thúc. Replay bị cắt, thiếu bước hoặc sai cấu hình không thể đạt.

Comparator so trạng thái, action và collision risk chính xác, với dung sai
tuyệt đối `1e-9` cho giá trị số. Nó cũng so yêu cầu AEB/MRM và trạng thái
gửi lệnh của controller khi đi qua nhánh override. Trạng thái gửi lệnh là
kết quả từ controller, còn diễn biến xe được đánh giá ở báo cáo mô phỏng.
Mã thoát `0` là PASS, `1` là mismatch, `2` là invalid hoặc incomplete và
`3` là lỗi thực thi. Khi có sai khác, `first_mismatch.json` lưu tick đầu tiên.

## Kết quả ngày 2026-11-19

Các lượt mô phỏng dùng CARLA build `edf3e9f5c`, Town02, trời quang, seed 42
và suy luận CPU. Manifest ghi HEAD `fa83772` và `git_dirty: true` vì lượt
chạy dùng bản nâng cấp đang kiểm tra trên máy.

- **HardBrake, 20 giây:** kịch bản PASS, 0 va chạm, khoảng hở nhỏ nhất
  5,78 m. Replay PASS **859/859** bước, gồm 59 bước warm-up. Báo cáo tại
  `logs/python-safety-20261119-005748/`.
- **HardBrake mất LiDAR và radar trong 4 giây đầu:** kịch bản PASS,
  0 va chạm, khoảng hở nhỏ nhất 5,01 m. Ghi nhận ODD VIOLATION và MRM.
  Replay PASS **859/859** bước. Báo cáo `fault-scenario.json` và
  `fault-comparison.json` tại `logs/python-safety-check-20261119-005955/`.
- **Runtime, 5 giây, low-memory, async-stable, 0 NPC:** replay PASS
  **200/200** bước. Runtime đạt 40,01 Hz và cleanup được xác minh. Tiêu
  chí runtime tổng thể FAIL vì ba chỉ số vượt ngưỡng: p99
  `safety_control` 32,81 ms (ngưỡng 25 ms), p95 suy luận 75,83 ms
  (ngưỡng 50 ms) và p95 tuổi kết quả 157 ms (ngưỡng 150 ms). Báo cáo
  `runtime.json` và `runtime-comparison.json` trong cùng thư mục với ca fault.

Không có tick bị rơi hoặc sai khác trong ba lượt replay. Fixture offline
đối chiếu đủ 29 bước. Sau refactor, 210 self-check và 321 unit test offline
đều đạt trên Windows, cùng các kiểm tra Ruff trên file Python đã đổi.

## Phạm vi đánh giá

Replay hiện so bản Python chạy lại với quyết định đã ghi trên cùng đầu vào.
Báo cáo kịch bản và báo cáo runtime giữ tiêu chí nghiệm thu riêng. Các lượt
kiểm tra ngắn trong đợt này chưa thay thế ma trận nhiều seed, nhiều thời tiết
và phép đo overhead trong kế hoạch. Phần C++ và chạy đối chiếu `cpp-shadow`
được hoãn theo phạm vi đã chọn.

## Đối chiếu kế hoạch triển khai ngày 19-11-2026

| Mục trong kế hoạch | Tiến độ CARLA/Python | Bằng chứng hoặc phần còn chờ |
|---|---|---|
| T01 / D01, baseline | Đạt trên bản Python dùng làm mốc | SHA-256 của cả 3 file gốc trong fixture khớp Git tại `fa83772`. Baseline ghi cấu hình, 210 self-check, 310 unit test và đầu ra của 29 bước. Kế hoạch ban đầu tham chiếu `2abeb69`. |
| T02 / D02, tách FSM | Đạt tiêu chí hồi quy Python | 7 episode đi qua public `update()` và reset. Quyết định và lý do chuyển trạng thái khớp fixture gốc. Có test riêng ngay hai phía ngưỡng khoảng cách và TTC. |
| T03 / D03, replay | Đạt tiêu chí phát lại đầy đủ trên các ca đã thử | Ghi warm-up, DRIVE, trạng thái phanh và cấu hình. Đếm theo từng episode từ JSONL. Chẩn đoán chi tiết ở mục 5.1 của kế hoạch vẫn còn thiếu TTC riêng từng nguồn và ID track trong bản replay. |
| T04–T06 / D04–D06 | Hoãn phần C++ | Comparator hiện đối chiếu Python. Phép so Python/C++, build hai hệ điều hành và shadow trong CARLA còn chờ. |
| T07–T08 / D07, đánh giá | Có bằng chứng Python cho hai ca HardBrake và một cấu hình runtime | Còn ma trận nhiều seed, hai thời điểm fault, các ca âm tính, độ lặp lại và phép đo overhead của shadow. |

### Lỗi tìm thấy và cách xác minh

- Bỏ tick cuối của capture 9 tick từng cho kết quả 9 đã ghi và 0 bị rơi.
  Reader hiện đếm 8 đã ghi, 1 bị rơi và kết luận `INCOMPLETE`. Khi thiếu
  tick giữa episode, reader chỉ so đến trước chỗ thiếu rồi tiếp tục so từ
  lần reset tiếp theo. Thiếu trường `diagnostics` trả về `INVALID`.
- Runtime từng bỏ trống danh sách mô hình YOLO. Harness từng ghi tên trọng
  số cấu hình ngay cả khi detector nạp một file tối ưu khác. Hai nơi hiện lấy
  đường dẫn và SHA-256 từ detector đã nạp. Đọc Git ở dạng byte cũng sửa lỗi
  giải mã diff tiếng Việt trên Windows. Manifest ghi thêm hash các module
  Python chưa commit.
- Vòng chạy headless từng vẽ bounding box mỗi tick và dựng dashboard mỗi
  frame hiển thị. Lượt kiểm tra 200 tick đặt kiểm tra lỗi ngay tại hai hàm
  vẽ, hoàn thành đủ thời lượng và xác minh cleanup. Runtime tính tuổi của kết
  quả camera đã lưu tại lúc sử dụng, sau khi chờ cảm biến.

### Kết quả kiểm tra lại

Artifact của đợt kiểm tra nằm trong `logs/plan-audit-20261119/final/`.
Lượt live dùng Town02, seed 42, trời quang, CPU và working tree đang sửa.

| Phép thử | Kết quả | Artifact |
|---|---|---|
| Fixture offline | 29/29 bước khớp | `fixture-comparison.json` |
| HardBrake chạy lại, 20 s | PASS, 0 va chạm, khoảng hở bề mặt 5,77 m, phản ứng tại frame 5 khi hazard xuất hiện. Replay khớp 859/859 bước. | `brake.json`, `brake-comparison.json` |
| HardBrake mất LiDAR và radar | Phát lại capture đã ghi ở đợt trước, khớp 859/859 bước | `fault-comparison.json` |
| Runtime headless, 5 s, 0 NPC | Replay khớp 200/200 bước. Runtime FAIL, 40,095 Hz, cleanup đạt. | `runtime.json`, `runtime-comparison.json` |

Trên 200 tick runtime, p99 đọc cảm biến là 30,432 ms, p99 từ lúc có dữ liệu
đến sau khi gửi lệnh là 3,910 ms, p99 ghi replay là 0,357 ms. Ngưỡng
`safety_control` 25 ms áp dụng cho cả đoạn chờ và xử lý. Giá trị đo được là
32,546 ms. Trên 49 kết quả nhận thức được sử dụng, p95 suy luận là
65,776 ms (ngưỡng 50 ms). Tuổi kết quả p95 là 172 ms (ngưỡng 150 ms).
Đây là số đo một lượt ngắn trên CPU.

Kiểm tra sau sửa trên Windows đạt 210 self-check, 323 unit test theo danh
sách offline của CI và 29 test controller/cảm biến dùng thư viện CARLA.
Ruff đạt trên repo, Mypy đạt trên 4 module trong cấu hình của dự án.
