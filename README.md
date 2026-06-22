# Dish Analysis - 智能膳食分析系统

基于 YOLO 深度学习模型与 DeepSeek AI 的多智能体菜品识别与膳食推荐系统。用户可通过拍照识别菜品，获取营养成分分析、个性化膳食建议、过敏预警及饮食记录追踪。


## 功能特性

| 智能体 | 名称 | 职责 |
|--------|------|------|
| 小盘 | 膳食推荐师 | 根据 BMI、运动量、健康目标生成一日三餐膳食方案 |
| 小物 | 营养分析师 | 针对特定菜品给出个性化营养评价与建议 |
| 小记 | 记录分析师 | 分析饮食记录规律，关注用餐时间与营养均衡 |
| 小安 | 过敏预警师 | 识别菜品过敏原，保障用户饮食安全 |
| 小膳 | 总协调员 | 多智能体协作调度，支持 Function Calling 对话 |

### 核心功能
- 菜品识别：基于 YOLOv11m-cls / YOLOv26n 模型识别图片中的菜品（支持单目标与多目标）
- 营养查询：内置 400+ 常见菜品营养数据库（热量、蛋白质、碳水、脂肪、过敏原）
- 个性化推荐：结合用户 BMI、运动量、健康目标、过敏原生成定制建议
- 饮食记录：保存识别历史，支持按时间倒序查询与删除
- AI 对话：支持自然语言交互，自动调度小记/小物/小安/小盘协作


## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Flask + Flask-CORS |
| 深度学习 | Ultralytics YOLOv11m-cls / YOLOv26n |
| AI 模型 | DeepSeek Chat API (Function Calling) |
| 数据库 | SQLite |
| 数据存储 | Pandas (Excel 类别映射) |
| 环境管理 | python-dotenv |


## 项目结构

```
Dish Analysis/
├── agent.py              # AI 提示词构建模块（小盘/小物/小记/小安）
├── analyze.py            # Flask 主服务（API 路由、模型推理、数据库操作）
├── nutrition_lib.py      # 400+ 菜品营养数据库
├── records.db            # SQLite 饮食记录数据库
├── requirements.txt      # Python 依赖清单
├── Dockerfile            # Docker 容器配置
├── .env                  # 环境变量（DeepSeek API Key）
├── dish_mapping/         # YOLO 类别映射（class_names.xlsx）
├── model/                # YOLOv8 单目标识别模型（best.pt）
└── model_yolov26/        # YOLOv26 多目标识别模型（best.pt）
```


## API 路由一览

| 方法 | 路由 | 功能 |
|------|------|------|
| POST | `/recognize` | 单目标菜品识别 |
| POST | `/recognize_multi` | 多目标菜品识别 |
| POST | `/api/record/save` | 保存饮食记录 |
| GET | `/api/record/list` | 获取用户记录列表 |
| DELETE | `/api/record/delete` | 删除指定记录 |
| POST | `/api/xiaowu_advice` | 小物营养分析 |
| POST | `/api/xiaopan_advice` | 小盘膳食推荐 |
| POST | `/api/xiaoji_advice` | 小记记录分析 |
| POST | `/api/chat` | 多智能体对话（Function Calling） |


## 快速开始

### 1. 环境要求

- Python 3.9+
- pip

### 2. 克隆项目

```bash
git clone https://github.com/your-username/dish-analysis.git
cd dish-analysis
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 5. 启动服务

```bash
python analyze.py
```

服务将在 `http://0.0.0.0:5000` 启动。

### 6. Docker 部署（可选）

```bash
docker build -t dish-analysis .
docker run -p 5000:5000 --env-file .env dish-analysis
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


## 请求 / 响应示例

### 识别菜品

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

### 对话接口

**请求：**
```json
POST /api/chat
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
  "blocks": [{"type": "text", "data": "..."}],
  "actions": [
    {"label": "有帮助", "action": "helpful"},
    {"label": "换一个", "action": "retry"}
  ]
}
```


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


## 注意事项

- DeepSeek API Key 必须配置在 `.env` 文件中，否则相关 AI 接口将返回错误
- 模型文件 `best.pt` 需放置在 `model/` 和 `model_yolov26/` 目录下
- 类别映射文件 `class_names.xlsx` 需放置在 `dish_mapping/` 目录下
- 默认端口为 5000，可通过修改 `app.run()` 参数调整


## 作者

Zhourun-d
