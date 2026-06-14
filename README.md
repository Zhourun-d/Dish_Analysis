# Dish Analysis - 智能膳食推荐系统

基于 YOLO 深度学习模型和 DeepSeek AI 的菜品识别与个性化膳食推荐系统。

## 功能特性

- 菜品识别：使用YOLO模型识别图片中的菜品，输出菜品名称和识别置信度
- 营养分析：查询菜品的营养成分（热量、蛋白质、碳水、脂肪等）
- 小盘推荐：根据用户的BMI、运动量、健康目标、过敏原等信息，生成整体膳食建议
- 小物分析：针对具体菜品，结合用户健康状况给出个性化营养分析和食用建议

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.9+ | 后端开发语言 |
| Flask | Web服务框架 |
| YOLO (Ultralytics) | 菜品识别模型 |
| DeepSeek API | AI对话生成膳食建议 |
| Pandas | 数据处理 |

## 项目结构

```
Dish_Analysis/
├── analyze.py          # Flask服务主文件
├── agent.py            # AI提示词构建模块
├── nutrition_lib.py    # 菜品营养数据库
├── model/
│   └── best.pt         # YOLO菜品识别模型
├── dish_mapping/
│   └── class_names.xlsx # 类别ID与菜品名称映射
├── .env                # 环境变量配置
└── requirements.txt    # Python依赖包列表
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Zhourun-d/Dish_Analysis.git
cd Dish_Analysis
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API Key

在项目根目录创建.env文件，添加DeepSeek API Key：

```
DEEPSEEK_API_KEY=你的API密钥
```

### 4. 启动服务

```bash
python analyze.py
```

服务默认运行于 http://localhost:5000

## API接口

### 菜品识别

POST /recognize

请求：form-data，字段名image，值为图片文件

响应示例：
```json
{
  "success": true,
  "dish": {
    "name": "麻婆豆腐",
    "probability": 85,
    "calories": 120,
    "protein": 6.5,
    "carbs": 6.0,
    "fat": 8.0,
    "allergens": "大豆",
    "tips": "麻辣鲜香，注意钠含量"
  }
}
```

### 小盘膳食推荐

POST /api/recommend

请求体示例：
```json
{
  "bmi": 22.5,
  "bmiStatus": "正常",
  "activity": "中等",
  "goal": "减脂",
  "allergens": ["花生", "海鲜"]
}
```

### 小物品菜分析

POST /api/xiaowu_advice

请求体示例：
```json
{
  "dish": {
    "name": "麻婆豆腐",
    "calories": 120,
    "protein": 6.5,
    "carbs": 6.0,
    "fat": 8.0,
    "allergens": "大豆"
  },
  "userProfile": {
    "bmi": 22.5,
    "bmiStatus": "正常",
    "activity": "中等",
    "goal": "减脂",
    "allergens": ["花生"]
  }
}
```

## 环境变量

| 变量名 | 说明 |
|--------|------|
| DEEPSEEK_API_KEY | DeepSeek API密钥 |

## 支持的菜品

项目内置150+道常见菜品的营养数据，涵盖豆腐类、蔬菜类、肉类、海鲜类、主食类、汤品类、甜点类等。详见nutrition_lib.py中的NUTRITION_DATA字典。

## 许可证

MIT License

## 作者

Zhourun-d
