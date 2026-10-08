# ==================== 构建阶段 ====================
FROM python:3.9-slim AS builder

WORKDIR /build

# 安装编译依赖（个别依赖缺 wheel 时需要从源码构建）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 配置国内镜像源加速
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .

# 只下载依赖到临时目录
RUN pip install --no-cache-dir --target=/deps -r requirements.txt


# ==================== 运行阶段 ====================
FROM python:3.9-slim

WORKDIR /app

# 复制依赖
COPY --from=builder /deps /usr/local/lib/python3.9/site-packages/

# 复制代码与模型：model/、model_yolov26/、dish_mapping/ 均由 COPY 一并带入，
# .env、.venv、__pycache__ 等通过 .dockerignore 排除
COPY . .

# ultralytics 依赖的 OpenCV 与 PyTorch 所需的系统库，slim 基础镜像默认不含；
# 缺失时会在 import ultralytics 阶段报 libGL.so.1: cannot open shared object file
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 构建期校验必需文件，避免镜像构建成功、容器启动才报错
RUN test -f model/best.pt \
    && test -f model_yolov26/best.pt \
    && test -f dish_mapping/class_names.xlsx \
    || (echo "构建失败：缺少 model/best.pt、model_yolov26/best.pt 或 dish_mapping/class_names.xlsx" && exit 1)

EXPOSE 80

# 使用 gunicorn 运行（多进程，更稳定）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple gunicorn

# --preload：先在主进程完成模型加载再 fork，避免每个 worker 各加载一份模型
CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "2", "--preload", "--timeout", "120", "analyze:app"]
