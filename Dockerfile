# Sử dụng Python 3.11 Slim làm base image
FROM python:3.11-slim

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (C++ compiler cho FAISS/Numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Tạo user không phải root (UID 1000) theo chuẩn Hugging Face Spaces
RUN useradd -m -u 1000 user

# Copy requirements và cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY --chown=user:user . .

# Tạo các thư mục cần thiết và phân quyền cho user 1000
RUN mkdir -p /app/data/uploads /app/data/processed && \
    chown -R user:user /app

# Chuyển sang user non-root
USER user

# Expose cổng 7860
EXPOSE 7860

# Khởi chạy server FastAPI với Uvicorn dùng biến PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
