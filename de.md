Edge AI Challenge 2026 là cuộc thi học thuật dành cho sinh viên, tập trung vào nghiên cứu và phát triển các mô hình trí tuệ nhân tạo có khả năng xử lý hình ảnh và hỏi đáp trực quan trong điều kiện tài nguyên tính toán hạn chế.
Cuộc thi đặt ra bài toán xây dựng một hệ thống Visual Question Answering (VQA) trên thiết bị biên (Edge AI). Hệ thống cần có khả năng nhận diện các vật thể xuất hiện trong hình ảnh, hiểu ý định của câu hỏi bằng tiếng Việt và đưa ra câu trả lời tương ứng.
Mỗi đội thi cần phát triển hai thành phần học máy chính:
Xây dựng mô hình phát hiện và xác định vị trí các vật thể trong ảnh thông qua bounding box. Bộ dữ liệu bao gồm 7 lớp đối tượng:
- bottle: Chai.
- cup: Cốc.
- phone: Điện thoại.
- book: Sách.
- mouse: Chuột máy tính.
- box: Hộp.
- toy_car: Ô tô đồ chơi.
Xây dựng mô hình TextCNN để phân loại câu hỏi tiếng Việt thành 5 nhóm ý định:
- COUNT: Đếm số lượng vật thể.
- EXIST: Xác định sự tồn tại của vật thể.
- LIST: Liệt kê các vật thể xuất hiện trong ảnh.
- LOCATION: Xác định vị trí của vật thể.
- OUT_OF_SCOPE: Nhận diện câu hỏi nằm ngoài phạm vi bài toán.
Kết quả dự đoán từ hai mô hình được kết hợp với bộ xử lý logic để tạo ra câu trả lời theo định dạng quy định.
Cuộc thi gồm hai giai đoạn:
Vòng bảng – Kaggle Online Competition
Các đội thực hiện huấn luyện, tối ưu và đánh giá mô hình trong môi trường Kaggle. Hệ thống chấm điểm tự động dựa trên hai thành phần: mAP@0.5 của mô hình Object Detection và Accuracy của mô hình Intent TextCNN.
Mỗi đội nộp một tệp submission.csv duy nhất chứa kết quả dự đoán cho cả hai bài toán.
Vòng chung kết – Triển khai trên thiết bị Edge AI
Các đội vượt qua vòng bảng sẽ triển khai hệ thống hoàn chỉnh trên phần cứng ESP32-S3 N16R8 kết hợp camera OV7670. Toàn bộ quá trình xử lý hình ảnh, phát hiện vật thể và phân loại ý định phải được thực hiện trực tiếp trên thiết bị, không sử dụng máy tính, máy chủ hoặc dịch vụ đám mây để thực hiện suy luận.
Các mốc thời gian trên được thực hiện theo kế hoạch tổ chức của Ban Tổ chức. Mọi thay đổi (nếu có) sẽ được thông báo công khai đến các đội tham gia.
- Nghiên cứu và phát triển các mô hình phát hiện vật thể có kích thước nhỏ, độ chính xác cao.
- Xây dựng mô hình phân loại ý định câu hỏi tiếng Việt bằng TextCNN hoặc các mô hình học máy khác có khả năng triển khai trên ESP32-S3.
- Kết hợp thị giác máy tính và xử lý ngôn ngữ tự nhiên trong một hệ thống hỏi đáp trực quan.
- Tối ưu mô hình về độ chính xác, số lượng tham số, dung lượng lưu trữ và thời gian suy luận.
- Đánh giá khả năng triển khai các mô hình AI trên phần cứng nhúng có tài nguyên hạn chế.
Cuộc thi hướng đến việc khuyến khích các giải pháp AI không chỉ đạt hiệu quả trên dữ liệu mà còn có khả năng triển khai và hoạt động ổn định trên thiết bị thực tế.

QUY ĐỊNH THAM GIA – EDGE AI CHALLENGE 2026
Mỗi đội gồm từ 1 đến 4 thành viên. Mỗi sinh viên chỉ được tham gia một đội.
Các đội phải sử dụng đúng dữ liệu và định dạng nộp bài do Ban Tổ chức cung cấp.
Mỗi đội được nộp tối đa 4 bài trong mỗi 24 giờ.
Không được khai thác nhãn của tập dữ liệu kiểm tra ẩn.
Không được sử dụng nhiều tài khoản để thao túng bảng xếp hạng.
Dữ liệu bên ngoài và mô hình huấn luyện trước phải được khai báo đầy đủ (chỉ được dùng ở VÒNG CHUNG KẾT).
Các đội phải lưu giữ mã nguồn, notebook, cấu hình huấn luyện và mô hình để phục vụ công tác xác minh.
Ban Tổ chức có quyền kiểm tra tính hợp lệ và khả năng tái tạo kết quả của các đội.
Các đội vi phạm quy định có thể bị hủy kết quả hoặc loại khỏi cuộc thi.
Quyết định của Ban Tổ chức là quyết định cuối cùng đối với các vấn đề phát sinh trong quá trình tổ chức cuộc thi.