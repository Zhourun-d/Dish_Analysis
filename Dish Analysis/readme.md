# Dish Analysis - 智能膳食分析系统

基于 YOLO 深度学习模型与 DeepSeek AI 的多智能体菜品识别与膳食推荐系统。用户可通过拍照识别菜品，获取营养成分分析、个性化膳食建议、过敏预警及饮食记录追踪。

后端服务（Flask + SQLite），前端为微信小程序，不在本仓库内。


## 功能特性

| 智能体 | 名称 | 职责 | 调用方式 |
|--------|------|------|----------|
| 小盘 | 膳食推荐师 | 根据 BMI、运动量、健康目标生成一日三餐膳食方案 | 独立接口 / 对话调度 |
| 小物 | 营养分析师 | 针对特定菜品给出个性化营养评价与建议 | 独立接口 / 对话调度 |
| 小记 | 记录分析师 | 分析饮食记录规律，关注用餐时间与营养均衡 | 独立接口 / 对话调度 |
| 小安 | 过敏预警师 | 识别菜品过敏原，保障用户饮食安全 | 仅对话调度（本地判断，不调 AI） |
| 小膳 | 总协调员 | 多智能体协作调度，支持 Function Calling 对话 | `/api/chat` |

### 核心功能
- 菜品识别：基于 YOLO 分类模型（单目标）与 YOLO 检测模型（多目标）识别图片中的菜品
- 营养查询：内置 409 道菜品的营养数据（热量、蛋白质、碳水、脂肪、过敏原）
- 个性化推荐：结合用户 BMI、运动量、健康目标、过敏原生成定制建议
- 饮食记录：保存识别历史，支持按时间倒序查询与删除
- AI 对话：支持自然语言交互，自动调度小记/小物/小安/小盘协作


## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 后端框架 | Flask + Flask-CORS | 3.1.0 / 5.0.0 |
| 深度学习 | Ultralytics（YOLO 分类 / 检测） | 8.3.0 |
| 推理后端 | PyTorch（随 ultralytics 自动安装） | — |
| AI 模型 | DeepSeek Chat API（Function Calling） | deepseek-chat |
| 数据库 | SQLite | 标准库 |
| 数据读取 | Pandas + openpyxl（Excel 类别映射） | 2.2.0 / 3.1.5 |
| 环境管理 | python-dotenv | 1.0.0 |
| 生产部署 | gunicorn（仅 Docker 镜像内使用） | 构建时安装 |


## 项目结构

```
Dish Analysis/
├── analyze.py            # Flask 主服务（API 路由、模型推理、请求编排）
├── agent.py              # AI 提示词构建模块（小盘/小物/小记/小安）
├── services.py           # 共享基础设施：SQLite 连接与查询、DeepSeek 调用
├── nutrition_lib.py      # 菜品营养数据库（单目标库 + 多目标库）
├── records.db            # SQLite 饮食记录数据库（首次启动自动创建）
├── requirements.txt      # Python 依赖清单
├── Dockerfile            # Docker 容器配置
├── .dockerignore         # Docker 构建上下文排除项（挡住 .env 等）
├── .gitignore            # Git 忽略项（挡住 .env 等）
├── .env                  # 环境变量（DeepSeek API Key）——切勿提交到版本库
├── dish_mapping/         # YOLO 类别映射
│   └── class_names.xlsx  # 第一列为类别 ID，第二列为菜品名称
├── model/                # 单目标识别模型（YOLO 分类）
│   └── best.pt
└── model_yolov26/        # 多目标识别模型（YOLO 检测）
    └── best.pt
```

模块依赖方向为 `analyze → agent → services` 与 `analyze → services`，不存在循环导入；`analyze.py` 与 `agent.py` 共用 `services.py` 中的同一份数据库与 DeepSeek 调用实现。


## 环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

只有这一项是必需的。未配置时，所有涉及 AI 的接口（`/api/xiaowu_advice`、`/api/xiaopan_advice`、`/api/xiaoji_advice`、`/api/chat`）会返回错误，识别与记录接口不受影响。

> **安全提醒**：`.env` 内含真实密钥，请勿提交到 Git，也不要让它被打进 Docker 镜像。仓库已自带 `.gitignore` 与 `.dockerignore`，两者都排除了 `.env`，请勿删除这两条规则。


## 快速开始

### 1. 环境要求

- Python 3.9+
- pip

### 2. 准备模型与映射文件

以下文件体积较大，通常不纳入版本控制，需自行放置：

| 路径 | 说明 | 体积参考 |
|------|------|----------|
| `model/best.pt` | 单目标识别模型 | 约 20 MB |
| `model_yolov26/best.pt` | 多目标识别模型 | 约 6 MB |
| `dish_mapping/class_names.xlsx` | 类别 ID → 菜名映射 | 约 18 KB |

**这三个文件缺任何一个，服务都会在启动时直接报错退出**（`analyze.py` 在导入阶段就加载模型和 Excel）。Docker 构建阶段也会校验这三个文件是否存在，缺失会直接中断构建并给出提示。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> `ultralytics` 会自动拉取 PyTorch，**下载量在数百 MB 到数 GB 之间**，首次安装较慢。国内网络可先配置镜像源：
>
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```
>
> 若只需 CPU 推理，可先单独安装 CPU 版 torch 以减小体积：
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 4. 配置环境变量

见上节「环境变量」。

### 5. 启动服务

```bash
python analyze.py
```

服务将在 `http://0.0.0.0:5000` 启动（开发服务器，勿用于生产）。

### 6. Docker 部署（可选）

```bash
docker build -t dish-analysis .
docker run -p 5000:80 --env-file .env dish-analysis
```

访问 `http://localhost:5000`。

> **注意端口**：容器内由 gunicorn 监听 **80**（不是开发模式的 5000），所以映射要写 `-p 5000:80`。
>
> 镜像已包含 `model/`、`model_yolov26/` 与 `dish_mapping/`（由 `COPY . .` 带入），无需额外挂载；`.env`、`.venv`、`__pycache__`、`*.db` 等已由 `.dockerignore` 排除，容器内的 `records.db` 在启动时由 `init_db()` 自动创建。
>
> 由于 ultralytics 依赖的 OpenCV 需要图形库，最终镜像额外安装了 `libgl1`、`libglib2.0-0`、`libgomp1`，缺少它们会在导入阶段报 `libGL.so.1: cannot open shared object file`。


## API 路由一览

| 方法 | 路由 | 功能 | 必填参数 |
|------|------|------|----------|
| POST | `/recognize` | 单目标菜品识别 | multipart 表单 `image` |
| POST | `/recognize_multi` | 多目标菜品识别 | multipart 表单 `image` |
| POST | `/api/record/save` | 保存饮食记录 | JSON `userId`、`dishName` |
| GET | `/api/record/list` | 获取用户记录列表 | 查询串 `userId` |
| DELETE | `/api/record/delete` | 删除指定记录 | JSON `id` |
| POST | `/api/xiaowu_advice` | 小物营养分析 | JSON `dish`、`userProfile` |
| POST | `/api/xiaopan_advice` | 小盘膳食推荐 | JSON `bmi`、`goal` 等 |
| POST | `/api/xiaoji_advice` | 小记记录分析 | JSON `records` |
| POST | `/api/chat` | 多智能体对话（Function Calling） | JSON `message` |


## 请求 / 响应示例

### 单目标识别

**请求：**
```http
POST /recognize
Content-Type: multipart/form-data

image: <图片文件>
```

**响应：**
```json
{
  "success": true,
  "dishes": [
    {
      "name": "宫保鸡丁",
      "probability": 85,
      "calories": 160,
      "protein": 15,
      "carbs": 8,
      "fat": 8,
      "allergens": "坚果(花生)",
      "tips": "酸甜微辣，花生酥脆"
    }
  ]
}
```

### 多目标识别

**请求：** 同 `/recognize`。

**响应：** 与单目标结构相同，`dishes` 中最多 5 道菜，并在顶层多返回一个 `count` 字段：

```json
{
  "success": true,
  "dishes": [ { "name": "红烧肉", "probability": 78, "...": "..." } ],
  "count": 1
}
```

未检测到可信菜品时返回 HTTP 400 与 `{"success": false, "error": "未识别到可信菜品（置信度过低）"}`。

### 保存饮食记录

**请求：**
```json
{
  "userId": "user_1234567890_abc123",
  "dishName": "宫保鸡丁",
  "probability": 85,
  "calories": 160,
  "protein": 15,
  "carbs": 8,
  "fat": 8,
  "allergens": "坚果(花生)",
  "tips": "酸甜微辣，花生酥脆",
  "imageUrl": "http://tmp/xxxx.jpg",
  "recordedAt": "2026-06-19 12:30"
}
```

`recordedAt` 格式固定为 `YYYY-MM-DD HH:MM`；缺省或格式错误时，服务端会改用当前时间。

**响应：**
```json
{ "success": true, "recordId": 123 }
```

### 对话接口

**请求：**
```json
{
  "message": "帮我制定一个减脂食谱",
  "userId": "user_1234567890_abc123",
  "userProfile": {
    "bmi": "22.5",
    "bmiStatus": "正常",
    "activity": "中",
    "goal": "减脂",
    "allergens": ["花生"]
  },
  "history": []
}
```

**响应：**
```json
{
  "success": true,
  "content": "小盘说：根据您的BMI和减脂目标...",
  "blocks": [{ "type": "text", "data": "..." }],
  "actions": [
    { "label": "有帮助", "action": "helpful" },
    { "label": "换一个", "action": "retry" }
  ]
}
```


## 数据库结构

### records 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 自增主键 |
| user_id | TEXT | 用户唯一标识 |
| dish_name | TEXT | 菜品名称 |
| probability | INTEGER | 识别置信度（百分比） |
| calories | REAL | 热量（千卡/100g） |
| protein | REAL | 蛋白质（g/100g） |
| carbs | REAL | 碳水化合物（g/100g） |
| fat | REAL | 脂肪（g/100g） |
| allergens | TEXT | 过敏原信息 |
| tips | TEXT | 食用建议 |
| image_url | TEXT | 图片 URL |
| recorded_at | TIMESTAMP | 用餐时间 |
| created_at | TIMESTAMP | 记录创建时间 |

表在服务导入时由 `services.init_db()` 自动创建，无需手工建库；表已存在时会自动补上缺失的 `recorded_at` 字段。


## 多智能体协作流程（Function Calling）

```
用户提问
    ↓
小膳（总协调员）分析意图
    ↓
    ├── 需要营养分析 → 调用小物 → 返回营养建议
    ├── 需要记录分析 → 调用小记 → 返回饮食规律
    ├── 需要过敏检查 → 调用小安 → 返回过敏预警
    └── 需要膳食推荐 → 调用小盘 → 返回三餐方案
    ↓
小膳综合所有结果，生成最终回复
```

小膳可在一轮内并行调用多个工具；工具结果返回后，再进行一次不带工具的调用来综合成最终回复。


## 常见问题

**启动就报模型或 Excel 找不到**
三个数据文件没放齐，见「快速开始」第 2 步。

**AI 接口返回「DeepSeek API 密钥未配置」**
`.env` 未创建、未放在项目根目录，或键名不是 `DEEPSEEK_API_KEY`。

**识别结果置信度普遍偏低**
模型原始置信度经 `calibrated_confidence()` 分段校准后输出，属于设计行为。多目标模型准确率低于单目标模型，其过滤阈值已放宽到 30%。

**`/api/record/list` 返回的记录不是我存的**
该接口按 `userId` 过滤，前端需传一致的用户标识；不传则回落到 `default_user`。

**Docker 容器起来了但访问不通**
容器内监听的是 80 端口，确认映射写的是 `-p 5000:80`。

**端口怎么改**
开发模式改 `analyze.py` 末尾 `app.run()` 的 `port` 参数；Docker 模式改 `Dockerfile` 里 gunicorn 的 `-b` 参数。


## 已知问题

以下是目前仍存在、但不影响正常启动与使用的问题：

1. **用餐时间的时区口径不一致**
   写入记录时用的是 Python 的本地时间（`datetime.now()`），而 `services.get_user_records()` 的窗口过滤用的是 SQLite 的 `datetime('now')`（UTC）。在东八区，「最近 7 天」实际会多算 8 小时。统一存 UTC 或统一用 `localtime` 即可。

2. **删除记录接口不校验归属**
   `/api/record/delete` 只按 `id` 匹配，未校验 `user_id`，理论上可以删除他人的记录。建议加上 `AND user_id = ?`。

3. **`/api/chat` 会重复插入 system 消息**
   `call_deepseek_with_tools()` 直接原地修改传入的 `messages`，而两轮调用都传了 `system_prompt`，导致第二条 system 消息被再次插入到开头。对输出影响很小，但确实多余。

4. **Function Calling 场景下上下文不完整**
   `agent.call_xiaoji_for_records()` 传入的 `user_profile` 是空字典，小记在工具调用路径下看不到用户的 BMI/目标/过敏原；`agent.call_xiaowu_for_nutrition()` 也没有去取历史记录。这与直接调用 `/api/xiaowu_advice` 时的效果不一致。

5. **`/api/record/list` 的 `total` 是当页条数**
   返回的 `total` 等于本次查询返回的记录数，并非该用户记录总数，做分页判断时会不准。

6. **`build_xiaoan_prompt()` 未被使用**
   过敏判断走的是 `call_xiaoan_for_allergy()` 里的本地逻辑（响应更快，且不消耗 token），该提示词构建函数属于遗留代码。


## 注意事项

- 模型文件 `best.pt` 需分别放置在 `model/` 与 `model_yolov26/` 目录下
- 类别映射文件 `class_names.xlsx` 需放置在 `dish_mapping/` 目录下
- 营养数据分两套库（`NUTRITION_DATA` 208 条、`NUTRITION_DATA_MULTI` 223 条，去重后共 409 道菜），两库有 22 道重名菜且个别数值不一致；多目标识别查不到时会回落到单目标库
- 识别接口的置信度阈值由前端判断，后端不做拦截
- 开发服务器默认端口 5000，容器内为 80
- `.env` 已被 `.gitignore` 与 `.dockerignore` 排除，请勿移除这两条规则


## 作者

Zhourun-d
