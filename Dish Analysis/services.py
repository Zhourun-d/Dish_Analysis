"""
共享基础设施模块

存放被 Flask 服务（analyze.py）和提示词构建模块（agent.py）共同依赖的部分：
数据库连接与查询、DeepSeek API 调用。

为什么要单独拆出来:
    这些函数原先都定义在 analyze.py 中，而 agent.py 在模块顶层
    `from analyze import get_user_records, call_deepseek`。由于 analyze.py 顶层
    又要 `import agent`，两者构成循环导入。直接 `python analyze.py` 时因为脚本
    模块名是 __main__，`from analyze import` 会触发一次全新的重复导入从而侥幸通过
    （代价是 analyze.py 被执行两遍、YOLO 模型加载两次）；但 gunicorn 以
    `analyze:app` 加载时模块名就是 analyze，会直接抛出

        ImportError: cannot import name 'get_user_records'
        from partially initialized module 'analyze'

    导致容器无法启动。抽到本模块后依赖方向变为
    analyze -> agent -> services 与 analyze -> services，不再成环。

环境变量:
    DEEPSEEK_API_KEY: DeepSeek API 密钥（在项目根目录的 .env 文件中配置）
"""

import os
import sqlite3

import requests
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置常量 ====================
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DB_PATH = './records.db'

if not DEEPSEEK_API_KEY:
    print("警告: DEEPSEEK_API_KEY 环境变量未设置，请检查 .env 文件")
    print("提示: 在项目根目录创建 .env 文件，添加 DEEPSEEK_API_KEY=你的密钥")


# ==================== 数据库操作 ====================
def get_db_connection():
    """
    获取 SQLite 数据库连接

    返回:
        sqlite3.Connection - 数据库连接对象，row_factory 设置为 sqlite3.Row
                            使得查询结果可以通过列名访问（如 row['dish_name']）

    注意:
        每次使用后需调用 conn.close() 释放连接
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    初始化数据库表结构

    功能说明:
        创建 records 表（如果不存在），包含以下字段:
            - id: INTEGER PRIMARY KEY - 自增主键
            - user_id: TEXT NOT NULL - 用户ID（小程序端生成）
            - dish_name: TEXT NOT NULL - 菜品名称
            - probability: INTEGER - 识别置信度（百分比）
            - calories: REAL - 热量（千卡/100g）
            - protein: REAL - 蛋白质（g/100g）
            - carbs: REAL - 碳水化合物（g/100g）
            - fat: REAL - 脂肪（g/100g）
            - allergens: TEXT - 过敏原信息
            - tips: TEXT - 食用建议
            - image_url: TEXT - 图片URL
            - created_at: TIMESTAMP - 记录创建时间（自动生成）
            - recorded_at: TIMESTAMP - 记录时间（用户指定的用餐时间）

        兼容性: 如果表已存在但缺少 recorded_at 字段，自动添加

    调用时机:
        模块导入时自动调用（见文件末尾）
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            dish_name TEXT NOT NULL,
            probability INTEGER,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            allergens TEXT,
            tips TEXT,
            image_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 检查字段是否存在，若不存在则添加（兼容已有数据库）
    cursor.execute("PRAGMA table_info(records)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'recorded_at' not in columns:
        cursor.execute("ALTER TABLE records ADD COLUMN recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        print("数据库升级：添加 recorded_at 字段")
    conn.commit()
    conn.close()


init_db()


# ======= 查询用户历史记录的工具函数，用于智能体分析 =======
def get_user_records(user_id, days=7, limit=20):
    """
    查询用户最近 N 天的饮食记录

    参数:
        user_id: str - 用户唯一标识（从小程序缓存中的 userId 获取）
        days: int - 查询最近几天的记录，默认 7 天
        limit: int - 最大返回条数，默认 20

    返回:
        list[dict] - 记录列表，按 recorded_at 降序排列（最新在前），每个元素包含:
            - id: int - 记录ID
            - dish_name: str - 菜品名称
            - calories: float - 热量
            - protein: float - 蛋白质
            - carbs: float - 碳水化合物
            - fat: float - 脂肪
            - allergens: str - 过敏原
            - tips: str - 食用建议
            - recorded_at: str - 记录时间（格式化: "YYYY-MM-DD HH:MM"）
            - recorded_date: str - 记录日期（格式化: "YYYY-MM-DD"）

    示例:
        >>> get_user_records("user_123", days=7, limit=10)
        [
            {"id": 1, "dish_name": "宫保鸡丁", "calories": 160, "recorded_at": "2026-06-19 12:30", ...},
            ...
        ]

    调用场景:
        小盘推荐（build_xiaopan_prompt）: 获取用户最近饮食记录，避免重复推荐
        小物分析（build_xiaowu_prompt）: 获取用户最近饮食记录，提供个性化建议
        小记分析（build_xiaoji_prompt）: 获取用户最近饮食记录，分析饮食规律
        /api/chat 对话接口: 获取用户最近饮食记录，作为上下文信息
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, dish_name, calories, protein, carbs, fat, allergens, tips,
               strftime('%Y-%m-%d %H:%M', recorded_at) as recorded_at,
               strftime('%Y-%m-%d', recorded_at) as recorded_date
        FROM records
        WHERE user_id = ? AND recorded_at >= datetime('now', ?)   -- 去掉 'localtime'
        ORDER BY recorded_at DESC
        LIMIT ?
    ''', (user_id, f'-{days} days', limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ==================== DeepSeek 调用 ====================
def call_deepseek(prompt, system_prompt=None, temperature=0.7, max_tokens=800):
    """
    通用 DeepSeek API 调用函数

    功能说明:
        向 DeepSeek API 发送提示词，返回 AI 生成的响应内容。
        支持自定义系统提示词、温度和最大 token 数。

    参数:
        prompt: str - 用户提示词（构建好的完整提示词）
        system_prompt: str | None - 系统提示词（定义 AI 的角色和行为）
        temperature: float - 生成温度，控制输出的随机性，范围 0-1，默认 0.7
        max_tokens: int - 最大输出 token 数，默认 800

    返回:
        str - AI 生成的响应内容

    异常:
        Exception - 请求超时、网络错误、API 错误等

    调用场景:
        被 agent.py 中的 call_xiaoji_for_records、call_xiaowu_for_nutrition、
        call_xiaopan_for_meal_plan 调用
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        raise Exception("请求超时，请稍后重试")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API 请求失败: {str(e)}")
