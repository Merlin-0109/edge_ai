Mỗi đội thi cần phát triển hai thành phần học máy chính:

1.1. Phát hiện vật thể (Object Detection)
Xây dựng mô hình phát hiện và xác định vị trí các vật thể trong ảnh thông qua bounding box. Bộ dữ liệu bao gồm 7 lớp đối tượng:

bottle: Chai.
cup: Cốc.
phone: Điện thoại.
book: Sách.
mouse: Chuột máy tính.
box: Hộp.
toy_car: Ô tô đồ chơi.
1.2. Phân loại ý định câu hỏi (Intent Classification)
Xây dựng mô hình TextCNN để phân loại câu hỏi tiếng Việt thành 5 nhóm ý định:

COUNT: Đếm số lượng vật thể.
EXIST: Xác định sự tồn tại của vật thể.
LIST: Liệt kê các vật thể xuất hiện trong ảnh.
LOCATION: Xác định vị trí của vật thể.
OUT_OF_SCOPE: Nhận diện câu hỏi nằm ngoài phạm vi bài toán.
Kết quả dự đoán từ hai mô hình được kết hợp với bộ xử lý logic để tạo ra câu trả lời theo định dạng quy định.

5. Quy định sử dụng dữ liệu
Các đội được phép sử dụng dữ liệu do Ban Tổ chức cung cấp để huấn luyện và đánh giá mô hình.
Không được sử dụng dữ liệu Test để gán nhãn thủ công, tạo pseudo-label hoặc huấn luyện mô hình trực tiếp hay gián tiếp.
Không được khai thác nhãn Test ẩn hoặc can thiệp vào hệ thống chấm điểm.
Các đội phải bảo đảm khả năng tái tạo kết quả dự đoán từ mã nguồn và cấu hình huấn luyện đã cung cấp.

Các đội thực hiện huấn luyện, tối ưu và đánh giá mô hình trong môi trường Kaggle. Hệ thống chấm điểm tự động dựa trên hai thành phần: mAP@0.5 của mô hình Object Detection và Accuracy của mô hình Intent TextCNN.