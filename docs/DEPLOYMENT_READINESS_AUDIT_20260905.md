# Rà soát lỗi và mức sẵn sàng deploy — 2026-09-05

## Kết luận

Cập nhật sau batch được duyệt, 2026-09-05 sau 18:03 local: A07–A09 đã sửa trong
`modules/ego_control.py`, có 16 test controller và 68 regression PASS, cùng
179 self-tests PASS. Năm bước đọc lại lệnh trên actor CARLA đứng yên đạt yêu cầu;
probe vẫn giữ FAIL do cleanup chưa xác nhận ngay, lần kiểm tra độc lập sau đó
xác nhận actor đã được dọn. Chi tiết:
[Controller batch summary](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/controller_fault_guard_20260905_1757/SUMMARY.md>).
Các mô tả lỗi/phạm vi duyệt phía dưới là snapshot trước sửa; không xóa lịch sử.
Các nhóm khác và native crash vẫn OPEN; không phải đã hoàn tất toàn bộ audit.

**NO-GO cho release hoàn chỉnh theo plan 8 tuần.** Có thể tiếp tục sửa offline theo từng batch đã duyệt; chưa được bỏ qua G1/G2 để train hoặc tuyên bố chạy ổn định mọi map.

Bản này tổng hợp **25 nhóm vấn đề nội bộ**: 12 nhóm đã tái hiện bằng fixture offline, 13 nhóm có bằng chứng code/báo cáo hoặc còn cần tái hiện sâu hơn. Một nhóm có thể chứa các biểu hiện cùng luồng; đây không phải 25 crash độc lập. Không thể bảo đảm đã tìm mọi latent bug chỉ bằng review.

| Phân loại theo yêu cầu | Kết quả | Cách xử lý |
|---|---|---|
| A — sửa trong cấu trúc/plan hiện tại | 25 nhóm bên dưới | Giữ chinh là orchestrator; sửa tại module sở hữu, regression rồi CARLA khi đủ điều kiện |
| B — bắt buộc phá cấu trúc/đổi plan | 0 trường hợp được chứng minh là bắt buộc | Chỉ có phương án dự phòng cần quyết định riêng; không mặc định viết lại |
| C — chưa thể sửa bằng code repo hiện có | 1 blocker native CARLA chưa có root cause | Điều tra engine/build/environment; không đồng nghĩa vĩnh viễn không sửa được |
| Giới hạn không phải bug | Dữ liệu chưa thu được và chứng nhận xe thật | Thu lại dữ liệu/đo lại; portfolio chỉ simulation, không tự biến thành chứng nhận L3 |

P0: chặn tin cậy safety/workflow/release; P1: lỗi chức năng/độ đúng cần xử lý trước gate tương ứng; P2: độ bền/observability/reproducibility. Không dùng mức độ này để tuyên bố hệ thống chạy xe thật an toàn.

## Phạm vi và phương pháp

Đọc plan Luna, setup AGENTS/QA/Code Reviewer/rules, code luồng chính và evidence dataset/model/crash. Kiểm tra AST + SHA-256 **111 Python sources ở root và modules**; semantic review tập trung sensor→perception→safety→control, training và acceptance. Không quét toàn bộ binary, dependency, tất cả ảnh bằng mắt; không phải penetration test hoặc chứng nhận malware-free.

Áp dụng [QA Engineer](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/Setup codex/codex-ultimate-setup/.codex/skills/qa-engineer/SKILL.md>) (reproduce, isolate, regression) và [Code Reviewer](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/Setup codex/codex-ultimate-setup/.codex/skills/code-reviewer/SKILL.md>) (where/why/repro/minimal fix). Template web trong setup không phải kiến trúc CARLA cần áp dụng.

Đợt review trước: 54 unittest PASS sau retry lỗi quyền TEMP và 179 self-tests PASS; cả log lỗi môi trường và retry được giữ tại [audit gốc](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/deployment_audit_20260905_1435/>). Không gộp thành một suite mới để tăng số test.

Preflight lúc khoảng 17:48 local:
- Hai test ego-control hiện có PASS, 179 self-tests PASS.
- Helper chạy thành công và ghi 14 quan sát lỗi; thêm sequence 3 bước xác nhận fault latch bị giảm phanh. **Exit 0 của helper không có nghĩa các lỗi đã được sửa.**
- 111 source hashes không đổi so với audit đã lưu.
- Không thấy tiến trình CARLA/Python tại snapshot 17:47:28; **CARLA tests NOT_RUN** ở đợt preflight này. Không tự restart.
- Chỉ lưu evidence/report/ledger; chưa sửa runtime, dependency, model hoặc dataset.

Bằng chứng và lệnh thực tế: [verification.json](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/fix_preflight_20260905_1748/verification.json>); [14 quan sát](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/fix_preflight_20260905_1748/reproductions.json>); [fault-latch sequence](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/fix_preflight_20260905_1748/fault_latch_reproduction.json>); [controller tests](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/fix_preflight_20260905_1748/controller_tests.log>); [self-tests](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/fix_preflight_20260905_1748/selftest.log>).

## A. Các nhóm sửa được trong kiến trúc hiện tại

Không cần bỏ YOLO, UFLDv2, radar/LiDAR hoặc viết lại planner để xử lý các nhóm này. Một số thay đổi cần interface giữa modules, nhưng đó là hoàn thiện hợp đồng đã có trong plan, không phải mất cấu trúc.

### A01 — Clock không hợp lệ được coi là mới

- Mức độ: P1; bằng chứng: Tái hiện; gate: W1.4.
- Vị trí chính: [modules/perception_contracts.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/perception_contracts.py>).
- Hiện trạng và hướng sửa nhỏ nhất: future/NaN/+Inf timestamp đều được chấp nhận. Chặn dữ liệu không hữu hạn, tương lai, sai thứ tự và reset clock; không dùng clamp về age=0 làm bằng chứng freshness.

### A02 — Mất LiDAR khi tắt radar không kích hoạt range-loss

- Mức độ: P0; bằng chứng: Tái hiện + code; gate: W1.3/W5.
- Vị trí chính: [modules/sensor_health.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/sensor_health.py>).
- Hiện trạng và hướng sửa nhỏ nhất: 4 lần LiDAR miss khi radar disabled vẫn range_redundancy_lost=false. SensorRig còn raise khi timeout trước bước cập nhật health/control. Sửa monitor và đường truyền loss tới ODD/MRM; test tác động lên control, không chỉ cờ health.

### A03 — Radar bị rút khoảng cách về cuối path

- Mức độ: P1; bằng chứng: Tái hiện + live liên quan; gate: W1.3/W5.
- Vị trí chính: [modules/road_geometry.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/road_geometry.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Radar tổng hợp ở 70 m được project thành 38.5 m khi path chỉ dài 38.5 m; safety dùng TTC=1.28 thay vì 2.33 s trong ví dụ thẳng closing=30 m/s. Sửa semantics finite-path projection trong road_geometry/active_safety, giữ raw range và nguồn TTC. Không bỏ radar-only hazard để giấu lỗi.

### A04 — Scenario có thể PASS khi thiếu bằng chứng

- Mức độ: P0; bằng chứng: Tái hiện; gate: W7, sửa trước khi dùng kết quả.
- Vị trí chính: [modules/scenario_acceptance.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/scenario_acceptance.py>).
- Hiện trạng và hướng sửa nhỏ nhất: KPI rỗng, không brake/evade nhưng reacted=true vẫn PASS ở static; NaN reaction cũng PASS. Clearance/collision phải được đo và hữu hạn; yêu cầu hành động phù hợp từng scenario và onset vật lý. Không ép mọi scenario có AEB=true.

### A05 — Learned lane không hợp lệ vẫn giành quyền

- Mức độ: P1; bằng chứng: Tái hiện; gate: W4.
- Vị trí chính: [modules/lane_source.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/lane_source.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Confidence NaN, centerline rỗng, frame đi ngược 5→1 vẫn chọn learned sau 5 quan sát. Kiểm tra finite, geometry thực, thứ tự và frame duy nhất; map fallback cho dữ liệu không an toàn.

### A06 — Throttle và brake cùng dương

- Mức độ: P1; bằng chứng: Tái hiện; gate: W6.
- Vị trí chính: [modules/vehicle_controller.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/vehicle_controller.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Input throttle=0.4/brake=0.01 tạo hai pedal cùng dương; điều kiện hiện tại chỉ triệt throttle khi brake>0.02. Chặn xung đột sau rate limit; bổ sung kiểm tra finite/range và regression chuyển AEB/MRM.

### A07 — Custom thiếu world âm thầm chuyển sang TM

- Mức độ: P0; bằng chứng: Tái hiện; gate: W1.3.
- Vị trí chính: [modules/ego_control.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_control.py>).
- Hiện trạng và hướng sửa nhỏ nhất: EGO_CONTROL_MODE=custom và world=None cho mode traffic_manager. Validate config khi khởi tạo, không mặc định TM cho cấu hình thiếu/sai; chỉ explicit traffic_manager được opt-in.

### A08 — Lỗi ghi log ngăn lệnh dừng an toàn

- Mức độ: P0; bằng chứng: Tái hiện; gate: W1.3.
- Vị trí chính: [modules/ego_control.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_control.py>).
- Hiện trạng và hướng sửa nhỏ nhất: run_step lỗi rồi print phát sinh UnicodeEncodeError: apply_control_calls=0. Khóa fault và yêu cầu brake trước diagnostics; lỗi console/hazard-light không được ngăn phanh. RPC thất bại vẫn phải báo thật, không thể hứa xe nhận lệnh khi server đã chết.

### A09 — Fault latch bị lệnh override sau làm yếu

- Mức độ: P0; bằng chứng: Tái hiện mới; gate: W1.3.
- Vị trí chính: [modules/ego_control.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_control.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Sequence tổng hợp: fault brake=1.0 → AEB brake≈0.3 → MRM brake≈0.1, dù mode vẫn custom_fault_safe_stop. Giữ mức phanh bảo thủ của fault latch, không làm mất SAFE_STOP hold; kiểm tra cả thứ tự ưu tiên và chu kỳ tiếp theo.

### A10 — Decision trace có thể làm dừng control

- Mức độ: P2; bằng chứng: Tái hiện, trường hợp biên; gate: W1.4.
- Vị trí chính: [modules/decision_trace.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/decision_trace.py>).
- Hiện trạng và hướng sửa nhỏ nhất: centroid là NumPy array gây ambiguous truth ValueError; runtime gọi trace trước apply control. LiDAR hiện trả list nên chưa chứng minh đây là lỗi live hiện tại. Chuẩn hóa record, cô lập lỗi observability; giữ ring buffer và đủ dữ liệu nguyên nhân đầu tiên.

### A11 — Mẫu số deadline-miss không nhất quán

- Mức độ: P2; bằng chứng: Tái hiện; gate: W1.4.
- Vị trí chính: [modules/pipeline_metrics.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/pipeline_metrics.py>).
- Hiện trạng và hướng sửa nhỏ nhất: max_samples=2 nhưng lifetime deadline_misses=3; báo samples=2/misses=3. Tách window/lifetime và duration/count; unavailable GPU không phải zero. Không suy tỷ lệ từ hai mẫu số khác nhau.

### A12 — Runtime report có thể PASS dù thiếu phép đo

- Mức độ: P0; bằng chứng: Tái hiện; gate: W1.4.
- Vị trí chính: [modules/runtime_report.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/runtime_report.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Fixture không collision sensor, sync stats, GPU metrics và không di chuyển vẫn PASS nếu các số latency giả lập đẹp. Missing phải UNKNOWN/FAIL theo gate áp dụng; cần motion ở clear-driving, unique output FPS, model/fault/cleanup evidence. Đây là fixture, không phải một chuyến CARLA PASS.

### A13 — RGB cũ có thể fusion với LiDAR mới

- Mức độ: P1; bằng chứng: Code; gate: W1.4/W6.
- Vị trí chính: [chinh.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/chinh.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Async giữ RGB cache nhưng submit cùng point cloud hiện tại; bundle/health dùng frame/timestamp của vòng hiện tại, thời điểm dequeue thay capture. Giữ frame/time từng sensor; match hoặc motion-compensate đúng hợp đồng, không coi tên same_frame_point_cloud là bằng chứng đồng bộ.

### A14 — Đồng hồ async vẫn dùng dt cấu hình cố định

- Mức độ: P1; bằng chứng: Code + live liên quan; gate: W1.4/W6.
- Vị trí chính: [chinh.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/chinh.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Tracker/AEB/PID/MRM dùng FIXED_DELTA dù delivery async có thể nhảy frame; report dùng frame_count/FPS làm simulated_duration. Lấy simulation timestamps cho động học/age, monotonic cho latency. Trace live có bước nhảy frame; chưa xác định phần đóng góp vào closing-speed outlier.

### A15 — Lane source hiển thị không đồng nghĩa path điều khiển

- Mức độ: P1; bằng chứng: Code; gate: W4/W6.
- Vị trí chính: [modules/ego_driving_stack.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_driving_stack.py>).
- Hiện trạng và hướng sửa nhỏ nhất: chinh chọn learned path cho safety nhưng EgoDrivingStack tự dựng map path khác; turn intent/origin có thể khác. map_lane_estimate lấy offset từ điểm đầu path thường ở (0,0), không phải sai số làn GT. Dùng một hợp đồng path/source và đo offset thật; không viết lại global planner.

### A16 — Traffic semantics còn dùng GT runtime

- Mức độ: P1; bằng chứng: Code / thiếu qualification; gate: W3/W6.
- Vị trí chính: [modules/ego_driving_stack.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_driving_stack.py>).
- Hiện trạng và hướng sửa nhỏ nhất: run_step gọi get_traffic_light_state khi respect_traffic_controls=true, trái yêu cầu GT chỉ validation/debug. Tách rõ debug GT, vision/unknown policy; StopSign cần liên kết biển/route thay vì chỉ cooldown toàn cục. Thiếu classifier đã nghiệm thu không được gọi là perception đạt chuẩn.

### A17 — Taxonomy model mới khác runtime; NMS chéo class

- Mức độ: P1; bằng chứng: Code, nguy cơ khi promote; gate: W3.
- Vị trí chính: [modules/object_tracking.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/object_tracking.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Runtime mặc định COCO IDs nhưng dataset mới 0..7; đổi checkpoint đơn thuần có thể bỏ/mislabel class. NMSBoxes lần cuối dùng chung mọi class có thể loại VRU chồng xe. Bind taxonomy theo artifact; test class-aware merge. Không nói baseline COCO hiện tại tự nó sai.

### A18 — Lane worker có thể giữ object output

- Mức độ: P1; bằng chứng: Code + latency live FAIL; gate: W6.
- Vị trí chính: [modules/neural_perception.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/neural_perception.py>).
- Hiện trạng và hướng sửa nhỏ nhất: _neural_worker đợi lane_future.result trước publish object; GPU jobs lane/object không thực sự một lịch serial chung. shutdown(wait=True) có thể kéo dài. Tách publish/cadence công bằng, bounded ownership/shutdown và đo unique FPS; không gọi 10 Hz RGB debug là đạt 20 FPS.

### A19 — Training chưa enforce approval/hash G2

- Mức độ: P0; bằng chứng: Code xác nhận; gate: W3.1, trước mọi train.
- Vị trí chính: [train.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/train.py>).
- Hiện trạng và hướng sửa nhỏ nhất: train.py kiểm path/fraction rồi YOLO(...), không kiểm exact approved dataset hash; traffic trainer cũng chưa có shared G2 gate. Builder/review flag không bảo vệ việc gọi trainer trực tiếp/resume. Fail closed trước model/GPU; tất cả entry point dùng cùng preflight. B05 MITIGATED trong ledger cũ không chính xác.

### A20 — Promotion/evaluation chưa đủ điều kiện

- Mức độ: P1; bằng chứng: Code + model report; gate: W3.
- Vị trí chính: [train.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/train.py>).
- Hiện trạng và hướng sửa nhỏ nhất: baseline mặc định -1, latency là mean evaluator chứ không runtime p95, promoted có thể true dù không copy artifact; posttrain test được gọi mỗi run. Tách eligibility/qualification/deployment, matched baseline/protocol và final holdout; copy/version/select an toàn. Candidate hiện mAP50≈0.224915, chưa promote, không đạt 0.75.

### A21 — Harness không hoàn toàn cùng đường runtime

- Mức độ: P1; bằng chứng: Code; gate: W1.3/W7.
- Vị trí chính: [run_scenarios.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/run_scenarios.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Scenario/evaluate/collector vẫn có ego.set_autopilot(True), lifecycle/sync còn trùng; bản vá chinh không bao phủ chúng. Unknown scenario có thể bị bỏ nếu vẫn còn tên hợp lệ, limit/denominator/output/exit cần fail-closed. Share modules và test đủ planned IDs, cleanup, exit; không chạy lại biến thể native đã crash mà không có giả thuyết mới.

### A22 — Provenance chưa chặn toàn bộ unsafe load/download

- Mức độ: P1; bằng chứng: Code / security risk; gate: W1/W3/W8.
- Vị trí chính: [modules/object_tracking.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/object_tracking.py>).
- Hiện trạng và hướng sửa nhỏ nhất: YOLO(path) được gọi trước hash; hash chỉ ghi lại, không đối chiếu allowlist approved. Existing Ultralytics load/AMP/export có đường unsafe deserialization hoặc dependency/model download cần khóa và test offline; MiDaS hub là đường legacy, không phải mặc định chinh. Pin/hash/audit trước execute; không kết luận có virus và không cài gì ở phiên này.

### A23 — Radar/LiDAR/fusion chưa đủ phủ hình học

- Mức độ: P1; bằng chứng: Code / cần tái hiện riêng; gate: W5.
- Vị trí chính: [modules/radar_processor.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/radar_processor.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Radar transform mới thiên về translation, approval dấu velocity chưa bind đủ build/pose/processor; LiDAR ground/height/distance filters cố định cần test vật thấp, slope/curve. Track max_age theo tick và radar update cần kiểm expiry/covariance/ego-motion. Đây là review backlog, chưa chứng minh từng nhánh đã gây tai nạn/crash.

### A24 — Replay có rủi ro ghi đè và thiếu integrity

- Mức độ: P2; bằng chứng: Code; gate: W1.4.
- Vị trí chính: [modules/replay_io.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/replay_io.py>).
- Hiện trạng và hướng sửa nhỏ nhất: Append manifest + frame filename tái sử dụng có thể lẫn/ghi đè khi frame reset. Cần session ID, schema/calibration/hash, kiểm record hỏng và storage bound. Replay chỉ component benchmark, không thay live safety deadline; không xóa dữ liệu cũ.

### A25 — CI và tài liệu có khoảng trống coverage

- Mức độ: P2; bằng chứng: Code / bằng chứng; gate: W1/W8.
- Vị trí chính: [docs/BUG_LEDGER.md](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/docs/BUG_LEDGER.md>).
- Hiện trạng và hướng sửa nhỏ nhất: CI chỉ gọi selftest (label cũ 106), chưa chạy các test mới. Ledger còn kết luận autopilot/ghost và B05 quá mạnh; baseline chưa đủ RAM/effective override/untracked hashes để khẳng định reproducible full release. Bổ sung coverage/manifest và đính chính theo thời gian, không xóa lịch sử.

## B. Phương án có thể làm đổi baseline/phạm vi — chưa triển khai

Chưa có bằng chứng bắt buộc phải phá kiến trúc. Chỉ trình duyệt riêng nếu cách sửa trong A và chẩn đoán G1 không đủ:

1. Đổi CARLA custom build/Python pair/driver hoặc OS: thay baseline môi trường, cần compatibility + security review, cài riêng và rollback; không bảo đảm hết crash trước khi đo lại.
2. Chấp nhận profile giới hạn map/FPS/resolution thay full target: thay phạm vi acceptance. CPU/10 Hz có thể là debug profile nhưng không được ghi PASS cho gate 20 FPS. Không tự hạ ngưỡng.
3. Dùng GPU/máy khác hoặc tách render/inference: thay deployment topology và chi phí; chỉ sau benchmark chứng minh bottleneck, không phải giải pháp đã xác minh cho access violation.

## C. Blocker ngoài khả năng sửa trực tiếp bằng Python hiện tại

**C01 — Native CARLA access violation, OPEN/P0.** Crash XML mới nhất được đọc: UE4CC-Windows-919C5FF84503E2264747D2B2268CB7BD_0000; local 14:29:34, PID 16324, SecondsSinceStart=590. Lỗi `EXCEPTION_ACCESS_VIOLATION reading address 0x00000001e30d0168`; UE4 4.26.2, D3D12, RTX 4070 Laptop GPU, driver 592.00. Metadata/hash: [output/verification/deployment_audit_20260905_1435/crash_metadata.json](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/deployment_audit_20260905_1435/crash_metadata.json>).

Artifact này không chứng minh OOM, Intel rendering, driver là root cause hay chỉ autopilot gây lỗi. Runtime custom đã chạy được một số lượt ngắn rồi lại crash; bỏ implicit TM chưa đóng G1. Cần minimal reproduction + correlated native logs/stack và lần thử có kiểm soát; nếu phải thay build/driver, trình duyệt phương án B1. Khi server chết, Python không thể đảm bảo lệnh phanh được thực thi.

Hai giới hạn cần ghi khi deploy, không tính là hai bug code:
- Không thể phục hồi chính xác chi tiết ảnh/depth/measurement chưa được ghi, hoặc biến test holdout đã dùng thành chưa từng được nhìn thấy chỉ bằng đổi tên. Cần recapture/new holdout với provenance.
- CARLA portfolio không phải chứng nhận ADAS/L3 hay bằng chứng sẵn sàng xe thật. “Mọi map” chỉ được dùng cho danh sách map thực sự nghiệm thu; physically impossible stopping distance không thể sửa bằng cờ AEB.

## Đính chính evidence live và dataset

- Lượt clear custom 25 s đã chạy 1,000 vòng, khoảng151.3 m, zero recorded collisions và server còn sống; safety p99≈31.75 ms, perception p95≈114.08 ms nên **FAIL latency**. [Live clear report](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/W1_2_default_custom_clear_Town02_20260905_143000/runtime_report.json>).
- Hazard smoke 20 s chỉ đi khoảng0.216 m; AEB=true không chứng minh dừng từ tốc độ cao với GT clearance/reaction đạt chuẩn. [Hazard report](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/W1_2_default_custom_hazard_Town02_20260905_143500/runtime_report.json>).
- Clear trace 25 s dừng ở980/1,000 vòng và timeout LiDAR; native crash C01 xác nhận còn blocker. [Runtime report](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/W1_2_clear_decision_trace25_Town02_20260905_145000/runtime_report.json>) và [decision trace](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/verification/W1_2_clear_decision_trace25_Town02_20260905_145000/decision_trace.json>).
- Radar `distance=38.5 m` trong trace là khoảng cách đã project, **không phải raw radar range**. A03 đã tái hiện một nguyên nhân làm cự ly ngắn giả. Closing33.22–63.23 m/s cần đo/đối chiếu riêng; chưa đủ căn cứ kết luận tất cả là ghost hoặc áp filter loại ngay. Giữ conservative geometry safety.
- V1 có26,250 ảnh; audit trước có0 exact decoded-RGB duplicate groups,2,109 near-hash candidate groups,3 cross-split suspects,1,551 sharpness flags. Candidate không đồng nghĩa duplicate đã xác nhận hoặc lệnh xóa.
- Label VRU occlusion/TrafficLight definition/road-line tag, StopSign train-domain/scale và đa dạng scene còn chưa đạt. V3 chưa có pilot hợp lệ; không được train lại V1. Dataset object/lane/traffic-state có gate riêng.

Nguồn dataset: [docs/DATASET_V2_ACCEPTANCE.md](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/docs/DATASET_V2_ACCEPTANCE.md>), [docs/dataset_label_v3_progress.md](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/docs/dataset_label_v3_progress.md>), [output/dataset_audit_20260905/REPORT.md](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/dataset_audit_20260905/REPORT.md>), [output/image_quality_20260905/SUMMARY.md](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/output/image_quality_20260905/SUMMARY.md>).

## Trạng thái theo plan

| Gate | Trạng thái có thể khẳng định |
|---|---|
| G0 | PARTIAL: inventory/evidence đã có, chưa đủ reproducible release manifest |
| G1 | BLOCKED: native moving/full-stack crash; short PASS không đủ gate |
| G2/G2a | BLOCKED: pilot/label/clean complete dataset chưa approved |
| G3 | Candidate cũ FAIL mAP gate; mới BLOCKED bởi G2 |
| G4 | Adapter tồn tại; lane accuracy/source-switch chưa nghiệm thu |
| G5 | Geometry/radar code tồn tại; còn A02/A03 và live qualification |
| G6 | Live latency đã FAIL; chưa đạt unique20 FPS |
| G7 | Chưa đủ matrix; acceptance phải sửa trước khi tin PASS |
| G8 | BLOCKED; chưa được release hoàn chỉnh |

## Batch đầu tiên đề nghị duyệt: A07–A09

Phạm vi chỉ [modules/ego_control.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/modules/ego_control.py>) và [test_ego_control.py](<C:/Users/Admin/OneDrive/Desktop/Carla Simulator/Self-Driving-Perception/test_ego_control.py>), cùng evidence/ledgers. Không đổi chinh, sensor, radar, train hoặc threshold.

1. Reject cấu hình custom thiếu world và mode thiếu/sai; chỉ explicit TM được bật.
2. Khóa fault, yêu cầu manual full brake trước diagnostics; log/hazard-light failure không ngăn safety command.
3. Không cho weaker AEB/MRM giảm fault-latched brake; vẫn giữ SAFE_STOP hold và báo thật nếu control RPC thất bại.
4. Viết test thất bại trước: Unicode/broken console, lỗi diagnostic, mode sai/thiếu, chu kỳ sau fault, ưu tiên override và explicit TM compatibility. Chạy targeted → affected regressions → selftests → review diff.
5. Live test chỉ sau khi có server sẵn sàng và scope được duyệt; lưu log mới. Offline PASS không đóng native G1.

**Trạng thái: AWAITING_APPROVAL, chưa sửa runtime.** Sau batch này quay lại W1.2/G1; tiếp tục từng batch sensor/frame và acceptance, không triển khai 25 nhóm trong một patch. G2 training preflight phải xong trước bất kỳ train nào.

## Đoạn limitations có thể dùng trong report deploy hiện tại

> Đây là ADAS research prototype chỉ chạy trong CARLA. Một số lượt chạy ngắn đã thành công, nhưng native crash chưa được giải quyết và yêu cầu latency chưa đạt. Dataset/model và toàn bộ scenario/fault/weather/map matrix chưa được nghiệm thu. Kết quả self-test, AEB flag hoặc một dashboard không chứng minh khả năng tránh va chạm hay sẵn sàng xe thật. Chỉ công bố cấu hình, map, số liệu và các lượt thử đã có evidence; không công bố release full-plan tại thời điểm audit này.
