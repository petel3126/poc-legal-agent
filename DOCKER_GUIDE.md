# 🐳 Hướng dẫn Chạy Dự án với Docker (1-Click Run)

Dự án này đã được đóng gói hoàn chỉnh bằng **Docker** và **Docker Compose**. Bất kỳ ai khi nhận toàn bộ mã nguồn của dự án này đều có thể khởi chạy toàn bộ hệ thống (Web App + AI Engine + Neo4j Graph DB) chỉ với **1 lệnh duy nhất**.

---

## 🚀 Các bước khởi chạy cho người mới:

### 1. Chuẩn bị biến môi trường
Tạo file `.env` từ file mẫu `.env.example` và điền `GEMINI_API_KEY`:
```bash
cp .env.example .env
```

### 2. Khởi chạy toàn bộ hệ thống bằng Docker Compose
Mở Terminal tại thư mục gốc của dự án và chạy:
```bash
docker compose up --build -d
```

---

## 🌐 Các dịch vụ sẵn sàng hoạt động:

1. **Giao diện Web Chatbot & API Backend**:
   👉 Truy cập tại: **[http://localhost:8000](http://localhost:8000)**

2. **Giao diện Quản lý Đồ thị Neo4j (Browser UI)**:
   👉 Truy cập tại: **[http://localhost:7474](http://localhost:7474)**
   * **Username**: `neo4j`
   * **Password**: `legalpassword123`

---

## 🛑 Lệnh Dừng & Xem Logs

* **Xem logs trực tiếp**:
  ```bash
  docker compose logs -f web
  ```
* **Dừng hệ thống**:
  ```bash
  docker compose down
  ```
