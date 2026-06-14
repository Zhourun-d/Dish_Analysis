# 构建阶段
FROM python:3.9-slim AS builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 配置清华源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .

# 只下载依赖到临时目录
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

# 运行阶段
FROM python:3.9-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /deps /usr/local/lib/python3.9/site-packages/

# 复制代码（不包含模型文件）
COPY . .

# 创建模型目录（模型从云存储下载）
RUN mkdir -p /app/model

EXPOSE 80

# 使用 gunicorn 运行（多进程，更稳定）
RUN pip install gunicorn --no-cache-dir

CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "2", "--timeout", "120", "analyze:app"]