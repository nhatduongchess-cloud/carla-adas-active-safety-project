# Dataset label V3 — 2026-09-05

## Trạng thái

Đã triển khai bộ lọc nhãn và kiểm thử offline. **Chưa xác minh trên ảnh CARLA mới; chưa training-ready.** Hai lần pilot không thu được frame; lần thứ hai có crash report mới. Không đổi hoặc train lại dataset cũ, không promote checkpoint.

## Thay đổi

- `modules/dataset_labels.py`: projection hộp 3D có hướng, loại hộp cắt near-plane/camera nằm trong hộp; backproject depth vào hộp và yêu cầu semantic đúng lớp. Dùng bbox vùng hỗ trợ nhìn thấy, không dùng bbox amodal của vật bị che hoàn toàn. Ngưỡng pilot: ≥4 pixel, ≥10% diện tích vùng chiếu, dung sai metric 0,25 m, cạnh bbox ≥3 pixel, khoảng cách ≤80 m.
- `collect_carla_dataset.py`: dùng module riêng; actor box nhân transform actor một lần, environment box dùng world coordinates; TrafficLight lấy từng `get_light_boxes()` thay vì hộp cả actor/cột. Lưu semantic PNG, geometry, pose, FOV, nhãn bị loại và lý do; ghi policy/version vào record và manifest. Không thêm camera/sensor mới so với collector hiện có.
- Road-line mask lấy `carla.CityObjectLabel.RoadLines`, không dùng số cố định 6. API thực tế trên máy (`edf3e9f5c`) trả RoadLines=24, Poles=6. Vì vậy dữ liệu road-line cũ phải audit/thu lại trước khi fine-tune learned lane; chưa sửa các file mask cũ.
- Chặn append V3 vào annotation legacy/mixed-policy. Giữ hàm depth-filter legacy để đọc/audit dữ liệu cũ, nhưng collector mới không gọi nó.
- `build_yolo_adas_dataset.py`: chặn nhãn V3 nếu manifest chưa có review status `approved`. Integrity PASS không thay thế visual review; không sửa status thành approved để vượt gate.
- `review_label_pilot.py`: tạo overlay (xanh=giữ, cam=ứng viên bị loại), counts và số pixel road-line từ capture; không sửa source.
- `test_dataset_labels.py`: regression cho visibility, occlusion, geometry, policy guard, training-review gate và công cụ review.

API tham chiếu: [CARLA TrafficLight / BoundingBox](https://carla.readthedocs.io/en/latest/python_api/#carla.TrafficLight), [CARLA depth camera](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera). Bản CARLA custom trên máy vẫn cần đối chiếu geometry/depth bằng hình thật, không chỉ dựa vào tài liệu/API tồn tại.

## Kiểm thử simulator đã thực hiện

Khởi động DX11, RenderOffScreen, Low quality, không neural inference/training. Server readiness PASS, client và server cùng version `edf3e9f5c`. Pilot Town02/clear, 20 frame, seed4242, 0 NPC nền, 1 heavy vehicle, 1 two-wheeler, 2 walkers, tiếp cận TrafficLight.

- Lần 1: server mất phản hồi, collector exit1; 0 ảnh. Không có crash report mới đủ để gán nguyên nhân chính xác.
- Lần 2: chạy ngoài sandbox; server khởi động và chuyển map thành công, sau đó streaming connection refused; 0 ảnh. Crash report `UE4CC-Windows-A2B2B0B549357FCDFFA124A117BC20CA_0000`, 2026-09-05 06:28 giờ máy, `SecondsSinceStart=27`, `CrashType=Crash`, `EXCEPTION_ACCESS_VIOLATION reading address 0x00000001e30d0168`.
- Đây không phải bằng chứng OOM/D3D device-lost; báo cáo GPUCrash lúc 00:16 là crash cũ, không dùng giải thích lần mới.
- Dừng retry, không đổi driver/registry/TDR hoặc cài thêm dependency. Pilot root riêng: `carla_dataset_v1/label_v3_pilot_20260905`; chỉ có thư mục khởi tạo, không có dataset mới hợp lệ.

## Giới hạn còn lại

- Semantic + hỗ trợ geometry không bảo đảm đúng instance nếu hai vật cùng lớp chồng hình và gần nhau. Cần test/cân nhắc instance segmentation sau, không tuyên bố đã giải quyết mọi occlusion.
- Ngưỡng visibility và giả định radial depth cần kiểm tra trên bề mặt thật, cạnh ảnh, vật nghiêng, vật bị che một phần, nhiều khoảng cách/thời tiết. Không sửa threshold chỉ để có nhiều label.
- Nhãn nhỏ bị loại vẫn được ghi trong `rejected_objects` để review false-negative. Chưa cho phép dùng pilot làm negative/background train.
- Chưa sửa scene repetition, chưa thu StopSign train, chưa đo lại mAP/recall/latency, chưa chạy AEB. Town train trong lần khảo sát trước thiếu stop-sign mesh; cần kiểm chứng lựa chọn nguồn/map thay vì chuyển test sang train.
- Root cause của server access violation chưa xác định. Bước tiếp theo là isolate startup: ego + RGB trước, rồi semantic/depth, LiDAR/radar, cuối cùng thêm từng loại actor. Log mới đã bổ sung tên sensor trước spawn và mốc warm-up.

## Lệnh kiểm tra offline

Kết quả cuối lượt: compile các file sửa PASS; **16/16 unit test PASS** (12 label/review + 4 audit), **179 self-test PASS / 0 FAIL**. Unit test dùng dữ liệu tổng hợp không thay cho kiểm tra simulator; pilot thực tế vẫn FAIL như ghi ở trên. Đã rà lại các đường gọi collector → geometry/visibility → manifest → builder gate.

```powershell
.\.venvCarLa\Scripts\python.exe -m unittest test_dataset_labels test_audit_training_dataset -v
.\.venvCarLa\Scripts\python.exe selftest.py
```

Chỉ chạy lại collector sau khi kiểm tra server ổn định; không mở training cùng lúc. Khi có capture hợp lệ, chạy `review_label_pilot.py --dataset <pilot-root> --output <new-report-dir>` rồi review cả nhãn được giữ lẫn bị loại trước khi cho phép build YOLO.
