# Dataset V2: chuẩn bị dữ liệu trước, chưa GPU training

Yêu cầu mới nhất của người dùng (2026-09-05): bộ dataset hoàn chỉnh, ảnh rõ và không trùng trước khi train. Không chạy lại YOLO trên V1 để thay thế công việc này.

## Định nghĩa hoàn chỉnh

Dataset phải có đủ 8 lớp Person, Bicycle, Car, Motorcycle, Bus, Truck, TrafficLight, StopSign; nhãn đúng vật nhìn thấy, đủ dữ liệu gần/xa, route/góc nhìn và điều kiện môi trường. Chỉ đủ số lớp hoặc đủ 20.000 file không đồng nghĩa sẵn sàng training. Mục tiêu quy mô của plan cũ vẫn là khoảng 20.000 frame có ích sau lọc, không nhân bản frame để đạt số lượng.

Ưu tiên bộ object-detection trước; dataset lane và traffic-light-state là task khác cần mask/crop/state đúng tương ứng. Không gọi một folder YOLO là đã hoàn thành cả ba task.

## Các gate phải qua trước training

| Gate | Bằng chứng cần có |
|---|---|
| File và nhãn hợp lệ | Decode toàn bộ ảnh; nhãn hữu hạn, đúng taxonomy/bounds; không missing/orphan; nhãn rỗng được review là background thật |
| Nhãn nhìn thấy và đúng đối tượng | Review cả nhãn được giữ và bị loại, theo lớp/nguồn/khoảng cách; bbox đèn quanh cụm đèn, không cột; không nhãn người trên xe che khuất |
| Độ nét có ích | Ảnh gốc đủ chi tiết đối tượng cần học; review điểm nét thấp theo cùng nguồn/thời tiết, không dùng một số Laplacian làm chứng nhận |
| Không trùng chính xác | 0 duplicate theo SHA file và decoded RGB ở bản phát hành; khi ảnh trùng mà nhãn khác nhau phải review, không chọn nhãn ngẫu nhiên |
| Không lặp cảnh vô ích | Review pHash/near-duplicate và pose/route; quyết định giữ/loại có lý do, ảnh đại diện vẫn phủ vật nhỏ/VRU/tình huống thay đổi |
| Không rò split | Tách town/route/session trước, nhóm near-duplicate nghi ngờ xuyên split phải giải quyết; không chuyển Town05/Town10HD vào train |
| Coverage | Báo cáo số ảnh độc lập và box từng lớp, khoảng cách/kích thước/thời tiết; StopSign train không chỉ ảnh crop rất lớn; review riêng VRU và vật nhỏ |
| Nguồn/provenance | Ghi nguồn, license, taxonomy mapping, config, checksum, phiên bản nhãn và lịch sử loại mẫu |
| Phát hành đóng băng | Version output mới, manifest + checksum + báo cáo review; approval sau review thật; giữ raw/V1 nguyên vẹn |

Không thể chứng minh tuyệt đối “mọi ảnh đều không gần trùng” bằng một hash. Yêu cầu phát hành là không có duplicate đã xác nhận và không còn nhóm nghi vấn chưa xử lý theo quy trình đã ghi. Báo cáo luôn công bố giới hạn kiểm tra.

## Ảnh rõ nét không có nghĩa xóa hết thời tiết xấu

- Bộ clear phải đủ nét để đọc hình dạng vật thể, không motion blur vô ích, không zoom ảnh bé rồi gọi là HD.
- Mưa/sương mù vẫn là dữ liệu ODD cần có. Đánh giá chất lượng của nhóm này riêng, không loại hết vì tương phản thấp.
- Không dùng AI sharpen/generate để “phục hồi” chi tiết biển báo/đèn/VRU không tồn tại trong ảnh gốc rồi giữ nhãn GT cũ.
- Thu pilot ở độ phân giải native hiện có trước. Chỉ tăng resolution/JPEG quality sau khi xác minh CARLA ổn định; không ép 1080p/4K trên máy đang crash.

## Trình tự công việc

1. Audit bổ sung độ nét và near-duplicate toàn bộ V1, lưu hàng đợi review, không sửa/xóa ảnh nguồn. Công cụ mới: `scripts/dataset/audit_image_quality.py` và `modules/image_quality.py`.
2. Review các nhóm lớn và mẫu độ nét thấp theo lớp/thời tiết; xác định phần external nào có thể giữ sau kiểm tra nhãn, phần CARLA nào phải thu lại. Không suy ra toàn bộ dữ liệu tốt chỉ vì detector nhìn thấy Bus.
3. Xác minh collector V3 trên simulator: hiện còn blocker server access violation, pilot V3 có 0 ảnh. Cần smoke test sensor/actor từng bước trước khi thu mới. Đây là công việc bắt buộc để có dataset đúng, không phải tối ưu runtime phụ.
4. Thu pilot đa dạng có pose, đủ clear + degraded có chủ đích; kiểm tra bbox/mask/crop/visibility. StopSign ở town train cần nguồn mesh hoặc dữ liệu bổ sung hợp lệ; không lấy test làm train. Nếu tải nguồn ngoài: kiểm tra nguồn/license/security/checksum và hỏi người dùng trước tải/cài.
5. Thu/relabel thành V2 mới; chọn frame theo thay đổi vị trí/cảnh và đối tượng, không lấy mọi tick vào dataset. Lưu đủ raw/provenance để audit được nhãn.
6. Review → deduplicate theo nhóm → kiểm tra split/coverage → đóng băng dataset. Chỉ sau đó chạy GPU pilot rồi training đầy đủ.

## Cảnh báo về manifest cũ

`carla_dataset_v1/yolo_adas_v1/dataset_manifest.json` có `training_ready_8_class=true` từ builder cũ. Cờ đó chỉ phản ánh điều kiện có lớp của lần build, KHÔNG chứng minh nhãn/độ nét/đa dạng đạt yêu cầu mới. Không dùng cờ này để tự khởi chạy training.

V3 vẫn phải qua `require_training_review` và review status hợp lệ. Không thay status/threshold để bỏ chặn.

## Công cụ bổ sung lần này

```powershell
.\.venvCarLa\Scripts\python.exe -B -m unittest test_image_quality -v
.\.venvCarLa\Scripts\python.exe -B -u audit_image_quality.py --dataset carla_dataset_v1\yolo_adas_v1 --output output\image_quality_next
```

Output phải là thư mục mới ngoài dataset. Các file `images.jsonl`, `groups.json`, `report.json`, `near_review_*.jpg` chỉ là bằng chứng audit/hàng đợi review, không phải dataset mới hay danh sách được phép train.
