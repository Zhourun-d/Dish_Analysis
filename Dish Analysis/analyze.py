"""
菜品识别与膳食推荐服务

模块功能概述:
    1. 菜品识别: 使用 YOLO 模型识别图片中的菜品
    2. 小盘推荐: 根据用户健康数据生成整体膳食建议（融合小记、小安）
    3. 小物分析: 针对特定菜品给出个性化营养建议（融合小记、小安）
    4. 小记分析: 饮食记录分析（增加过敏检查）
    5. 对话接口: 支持 Function Calling 实现智能体协作

API 路由总览:
    POST   /recognize              - 菜品识别
    POST   /api/record/save        - 保存饮食记录
    GET    /api/record/list        - 获取记录列表
    DELETE /api/record/delete      - 删除记录
    POST   /api/xiaowu_advice      - 小物营养分析
    POST   /api/xiaopan_advice     - 小盘膳食推荐
    POST   /api/xiaoji_advice      - 小记记录分析
    POST   /api/chat               - 对话接口（Function Calling）

前端调用链路:
    小程序 index.js (拍照识别) -> /recognize -> YOLO 模型 -> 返回识别结果
    小程序 index.js (小物分析) -> /api/xiaowu_advice -> agent.build_xiaowu_prompt -> DeepSeek
    小程序 analyse.js (小盘推荐) -> /api/xiaopan_advice -> agent.build_xiaopan_prompt -> DeepSeek
    小程序 record.js (小记分析) -> /api/xiaoji_advice -> agent.build_xiaoji_prompt -> DeepSeek
    小程序 chat.js (对话) -> /api/chat -> Function Calling -> 多智能体协作

依赖模块:
    - os, tempfile: 文件处理
    - flask: Web 框架
    - flask_cors: 跨域支持
    - pandas: 读取 Excel 类别映射
    - ultralytics: YOLO 模型
    - requests: HTTP 请求
    - datetime, json: 工具库
    - nutrition_lib: 营养数据
    - agent: 提示词构建
    - services: 数据库操作与 DeepSeek 调用

环境变量:
    DEEPSEEK_API_KEY: DeepSeek API 密钥（在 .env 文件中配置）
"""

import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from ultralytics import YOLO
import requests
from datetime import datetime
import json

from nutrition_lib import get_nutrition, get_nutrition_multi
from services import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    get_db_connection,
    get_user_records,
)
import agent

# ==================== 应用初始化 ====================
app = Flask(__name__)
CORS(app)

# 加载 YOLO 模型（用于菜品识别）
model = YOLO("./model/best.pt")
multi_model = YOLO("./model_yolov26/best.pt")

# 加载类别映射（Excel: 第一列类别ID，第二列菜品名称）
# 文件路径: ./dish_mapping/class_names.xlsx
df = pd.read_excel('./dish_mapping/class_names.xlsx', header=None)
id_to_name = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


# ==================== 工具函数 ====================
def calibrated_confidence(raw_conf):
    if raw_conf < 0.3:
        return raw_conf * 0.166
    elif raw_conf < 0.6:
        return 0.05 + (raw_conf - 0.3) * (0.30 - 0.05) / (0.60 - 0.30)
    elif raw_conf < 0.8:
        return 0.30 + (raw_conf - 0.6) * (0.70 - 0.30) / (0.80 - 0.60)
    elif raw_conf < 0.9:
        return 0.70 + (raw_conf - 0.8) * (0.75 - 0.70) / (0.90 - 0.80)
    else:
        return 0.75 + (raw_conf - 0.9) * (0.90 - 0.75) / (1.0 - 0.9)


def build_dish_response(dish_name, probability, nutrition_data):
    """
    统一构建菜品响应对象

    参数:
        dish_name: str - 菜品名称
        probability: int - 置信度百分比
        nutrition_data: dict | None - 营养数据（从 get_nutrition 或 get_nutrition_multi 获取）

    返回:
        dict - 标准菜品对象
    """
    if nutrition_data:
        return {
            'name': dish_name,
            'probability': probability,
            'calories': nutrition_data['calories'],
            'protein': nutrition_data['protein'],
            'carbs': nutrition_data['carbs'],
            'fat': nutrition_data['fat'],
            'allergens': nutrition_data['allergens'],
            'tips': nutrition_data['tips']
        }
    else:
        return {
            'name': dish_name,
            'probability': probability,
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'allergens': '未知',
            'tips': f'暂未收录 {dish_name} 的营养数据'
        }


def check_api_key():
    """
    检查 DeepSeek API 密钥是否已配置

    返回:
        tuple[bool, str | None] - (是否有效, 错误信息)
            - 有效: (True, None)
            - 无效: (False, "错误描述")

    调用场景:
        在所有需要调用 DeepSeek API 的路由中首先调用此函数进行校验
    """
    if not DEEPSEEK_API_KEY:
        return False, "DeepSeek API 密钥未配置"
    return True, None


# ===== 保存记录的接口，接收 recordedAt =====
@app.route('/api/record/save', methods=['POST'])
def save_record():
    """
    保存饮食记录接口

    功能说明:
        将用户识别的菜品保存到数据库，包含营养信息和用户指定的用餐时间。

    请求格式:
        POST /api/record/save
        Content-Type: application/json
        {
            "userId": "user_1234567890_abc123",   // 用户唯一标识
            "dishName": "宫保鸡丁",                // 菜品名称
            "probability": 85,                    // 识别置信度（百分比）
            "calories": 160,                      // 热量（千卡/100g）
            "protein": 15,                        // 蛋白质（g/100g）
            "carbs": 8,                           // 碳水化合物（g/100g）
            "fat": 8,                             // 脂肪（g/100g）
            "allergens": "坚果(花生)",             // 过敏原信息
            "tips": "酸甜微辣，花生酥脆",           // 食用建议
            "imageUrl": "http://tmp/xxxx.jpg",    // 图片URL（临时路径）
            "recordedAt": "2026-06-19 12:30"      // 用户指定的用餐时间（可选）
        }

    返回格式:
        成功: {
            "success": true,
            "recordId": 123                       // 新插入的记录ID
        }
        失败: {
            "success": false,
            "error": "错误信息"
        }

    前端调用:
        小程序 index.js -> addRecord() -> confirmAddRecord() -> saveRecord()
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求数据为空'}), 400

    user_id = data.get('userId', 'default_user')
    dish_name = data.get('dishName', '')
    probability = data.get('probability', 0)
    calories = data.get('calories', 0)
    protein = data.get('protein', 0)
    carbs = data.get('carbs', 0)
    fat = data.get('fat', 0)
    allergens = data.get('allergens', '')
    tips = data.get('tips', '')
    image_url = data.get('imageUrl', '')

    # ----- 处理 recorded_at -----
    recorded_at_str = data.get('recordedAt', '')
    if recorded_at_str:
        try:
            # 前端传来的格式应为 "YYYY-MM-DD HH:MM"
            recorded_at = datetime.strptime(recorded_at_str, "%Y-%m-%d %H:%M")
        except ValueError:
            # 如果格式不对，使用当前时间
            recorded_at = datetime.now()
    else:
        recorded_at = datetime.now()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO records 
        (user_id, dish_name, probability, calories, protein, carbs, fat, allergens, tips, image_url, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, dish_name, probability, calories, protein, carbs, fat, allergens, tips, image_url, recorded_at))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return jsonify({'success': True, 'recordId': record_id})


# ===== 记录列表按 recorded_at 倒序 =====
@app.route('/api/record/list', methods=['GET'])
def get_records():
    """
    获取用户饮食记录列表接口

    功能说明:
        分页查询用户的饮食记录，按 recorded_at 倒序排列（最新在前）。

    请求格式:
        GET /api/record/list?userId=user_123&limit=20&offset=0
        参数:
            - userId: str - 用户唯一标识（必填）
            - limit: int - 每页条数，默认 50
            - offset: int - 偏移量，从第几条开始，默认 0

    返回格式:
        成功: {
            "success": true,
            "records": [
                {
                    "id": 123,
                    "dish_name": "宫保鸡丁",
                    "probability": 85,
                    "calories": 160,
                    "protein": 15,
                    "carbs": 8,
                    "fat": 8,
                    "allergens": "坚果(花生)",
                    "tips": "酸甜微辣，花生酥脆",
                    "recorded_at": "2026-06-19 12:30"
                }
            ],
            "total": 1
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 record.js -> loadRecords() -> 分页加载记录列表
    """
    user_id = request.args.get('userId', 'default_user')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, dish_name, probability, calories, protein, carbs, fat, allergens, tips, 
               strftime('%Y-%m-%d %H:%M', recorded_at) as recorded_at
        FROM records
        WHERE user_id = ?
        ORDER BY recorded_at DESC
        LIMIT ? OFFSET ?
    ''', (user_id, limit, offset))
    rows = cursor.fetchall()
    conn.close()

    records = [dict(row) for row in rows]
    return jsonify({'success': True, 'records': records, 'total': len(records)})


# ===== 删除记录的接口 =====
@app.route('/api/record/delete', methods=['DELETE'])
def delete_record():
    """
    删除饮食记录接口

    功能说明:
        根据记录ID删除指定的饮食记录。

    请求格式:
        DELETE /api/record/delete
        Content-Type: application/json
        {
            "id": 123    // 记录ID
        }

    返回格式:
        成功: {"success": true}
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 record.js -> deleteRecord() -> wx.request -> /api/record/delete
    """
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'success': False, 'error': '缺少记录ID'}), 400
    record_id = data['id']
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM records WHERE id = ?', (record_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': '记录不存在'}), 404


# ===== 识别接口(分类模型) =====
@app.route('/recognize', methods=['POST'])
def recognize():
    """
    菜品识别接口

    功能说明:
        接收前端上传的图片，使用 YOLO 模型识别菜品，
        返回识别结果和对应的营养信息。

    请求格式:
        POST /recognize
        Content-Type: multipart/form-data
        form-data:
            - image: 图片文件（JPEG/PNG 格式）

    返回格式:
        成功: {
            "success": true,
            "dish": {
                "name": "宫保鸡丁",              // 菜品名称
                "probability": 85,               // 置信度百分比（校准后）
                "calories": 160,                 // 热量（千卡/100g）
                "protein": 15,                   // 蛋白质（g/100g）
                "carbs": 8,                      // 碳水化合物（g/100g）
                "fat": 8,                        // 脂肪（g/100g）
                "allergens": "坚果(花生)",        // 过敏原信息
                "tips": "酸甜微辣，花生酥脆"       // 食用建议
            }
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 index.js -> analyzeNutrition() -> wx.uploadFile -> /recognize

    注意:
        置信度阈值 40% 以下视为识别失败，前端会提示"没认出这道菜"
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '未上传图片'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '图片文件名为空'}), 400
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    temp_file.close()
    try:
        results = model(temp_file.name)
        probs = results[0].probs
        pred_id = probs.top1
        dish_name = id_to_name[pred_id]
        raw_conf = probs.top1conf.item()  # 提取最高置信度

        conf = calibrated_confidence(raw_conf)
        confidence_percent = int(conf * 100)

        # 从单目标营养库查询
        nutrition = get_nutrition(dish_name)

        # 使用统一构建函数
        dish = build_dish_response(dish_name, confidence_percent, nutrition)

        # ===== 改为返回 dishes 数组 =====
        return jsonify({'success': True, 'dishes': [dish]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


# ===== 识别接口(检测模型) =====
@app.route('/recognize_multi', methods=['POST'])
def recognize_multi():
    """
    多目标菜品识别接口

    功能说明:
        使用目标检测模型识别图片中的多个菜品，
        返回最多5个检测到的菜品及其营养信息。

    请求格式:
        POST /recognize_multi
        Content-Type: multipart/form-data
        form-data:
            - image: 图片文件（JPEG/PNG 格式）

    返回格式:
        成功: {
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
                },
                ...
            ],
            "count": 2
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 index.js -> 多目标模式 -> wx.uploadFile -> /recognize_multi
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '未上传图片'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '图片文件名为空'}), 400
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    temp_file.close()
    try:
        # 多目标检测推理
        results = multi_model(temp_file.name)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return jsonify({'success': False, 'error': '未识别到任何菜品'}), 400

        detected_dishes = []
        seen = set()

        for box in boxes:
            cls_id = int(box.cls.item())
            raw_conf = float(box.conf.item())

            conf = calibrated_confidence(raw_conf)
            # 置信度阈值 30%（多目标模型准确率较低，阈值适当放低）
            if conf < 0.3:
                continue

            # 获取类别名称（模型直接输出中文）
            dish_name = multi_model.names[cls_id]

            # 去重
            if dish_name in seen:
                continue
            seen.add(dish_name)

            confidence_percent = int(conf * 100)

            # 从多目标营养库查询
            nutrition = get_nutrition_multi(dish_name)

            # 如果多目标库没有，尝试从单目标库查询（兜底）
            if nutrition is None:
                nutrition = get_nutrition(dish_name)

            dish = build_dish_response(dish_name, confidence_percent, nutrition)
            detected_dishes.append(dish)

            # 最多返回5个
            if len(detected_dishes) >= 5:
                break

        if not detected_dishes:
            return jsonify({'success': False, 'error': '未识别到可信菜品（置信度过低）'}), 400

        return jsonify({
            'success': True,
            'dishes': detected_dishes,
            'count': len(detected_dishes)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


# ===== 小物接口 =====
@app.route('/api/xiaowu_advice', methods=['POST'])
def xiaowu_advice():
    """
    小物营养分析接口

    功能说明:
        针对用户识别的特定菜品，结合用户健康档案和历史饮食记录，
        使用 DeepSeek API 生成个性化的营养分析建议。

    请求格式:
        POST /api/xiaowu_advice
        Content-Type: application/json
        {
            "dish": {
                "name": "宫保鸡丁",
                "calories": 160,
                "protein": 15,
                "carbs": 8,
                "fat": 8,
                "allergens": "坚果(花生)"
            },
            "userProfile": {
                "bmi": "22.5",
                "bmiStatus": "正常",
                "activity": "中",
                "goal": "减脂",
                "allergens": ["花生"]
            },
            "userId": "user_1234567890_abc123"    // 用于获取历史记录
        }

    返回格式:
        成功: {
            "success": true,
            "advice": "小物说：宫保鸡丁含有花生，与您的过敏原冲突..."
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 index.js -> startXiaowuAnalyse() -> wx.request -> /api/xiaowu_advice
    """
    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"success": False, "error": error_msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求数据为空"}), 400

    dish = data.get('dish')
    user_profile = data.get('userProfile', {})
    user_id = data.get('userId', 'default_user')  # 获取 userId

    if not dish or not user_profile:
        return jsonify({"success": False, "error": "缺少菜品或用户信息"}), 400

    # ----- 获取用户最近7天记录 -----
    records = get_user_records(user_id, days=7, limit=10)
    # 将记录摘要加入 user_profile，供提示词使用
    user_profile['recent_records'] = records
    # dish为列表，user_profile为嵌套列表，里面包含allergens和recent_records列表
    prompt = agent.build_xiaowu_prompt(dish, user_profile)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是小物，一位亲切专业的营养分析智能体。回答要简洁自然，不用任何markdown格式，不用表情符号。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 400
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        advice = result['choices'][0]['message']['content']
        advice = agent.clean_response(advice)
        return jsonify({"success": True, "advice": advice})
    except Exception as e:
        return jsonify({"success": False, "error": f"处理失败: {str(e)}"}), 500


# ===== 小盘接口 =====
@app.route('/api/xiaopan_advice', methods=['POST'])
def xiaopan_advice():
    """
    小盘膳食推荐接口

    功能说明:
        根据用户的健康档案（BMI、运动量、目标、过敏原）和历史饮食记录，
        使用 DeepSeek API 生成整体性的膳食推荐方案（含早午晚餐建议）。

    请求格式:
        POST /api/xiaopan_advice
        Content-Type: application/json
        {
            "bmi": "22.5",
            "bmiStatus": "正常",
            "activity": "中",
            "goal": "减脂",
            "allergens": ["花生", "海鲜"],
            "userId": "user_1234567890_abc123"    // 用于获取历史记录
        }

    返回格式:
        成功: {
            "success": true,
            "recommendation": "小盘说：根据您的BMI和减脂目标，建议早餐..."
        }
        失败: {"error": "错误信息"}

    前端调用:
        小程序 analyse.js -> startAnalyse() -> wx.request -> /api/xiaopan_advice
    """
    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"error": error_msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据为空"}), 400

    bmi = data.get('bmi', '--')
    bmi_status = data.get('bmiStatus', '')
    activity = data.get('activity', '--')
    goal = data.get('goal', '--')
    allergens = data.get('allergens', [])
    user_id = data.get('userId', 'default_user')

    # ----- 获取用户最近7天记录 -----
    records = get_user_records(user_id, days=7, limit=10)
    # 与小物同理
    prompt = agent.build_xiaopan_prompt(bmi, bmi_status, activity, goal, allergens, records)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业的膳食推荐师，名字叫小盘。请用简洁自然的语言给出膳食建议，不要有任何markdown格式标记。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        recommendation = result['choices'][0]['message']['content']
        recommendation = agent.clean_response(recommendation)
        return jsonify({"success": True, "recommendation": recommendation})
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


# ===== 小记接口（增加过敏检查，可选）=====
@app.route('/api/xiaoji_advice', methods=['POST'])
def xiaoji_advice():
    """
    小记记录分析接口

    功能说明:
        分析用户的历史饮食记录，关注用餐时间规律、营养均衡度、
        热量控制等，给出改进建议。

    请求格式:
        POST /api/xiaoji_advice
        Content-Type: application/json
        {
            "records": [
                {
                    "id": 1,
                    "dish_name": "宫保鸡丁",
                    "calories": 160,
                    "recorded_at": "2026-06-19 12:30"
                }
            ],
            "userProfile": {
                "bmi": "22.5",
                "bmiStatus": "正常",
                "activity": "中",
                "goal": "减脂",
                "allergens": ["花生"]
            },
            "userId": "user_1234567890_abc123"
        }

    返回格式:
        成功: {
            "success": true,
            "advice": "小记说：小记注意到您最近三餐时间比较规律..."
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 record.js -> startXiaojiAnalyse() -> wx.request -> /api/xiaoji_advice
    """
    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"success": False, "error": error_msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求数据为空"}), 400

    records = data.get('records', [])
    user_profile = data.get('userProfile', {})

    if not records:
        return jsonify({"success": False, "error": "暂无记录可分析"}), 400

    prompt = agent.build_xiaoji_prompt(records, user_profile)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是小记，一位亲切专业的饮食记录分析师。回答要简洁自然，不用任何markdown格式，不用表情符号。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        advice = result['choices'][0]['message']['content']
        advice = agent.clean_response(advice)
        return jsonify({"success": True, "advice": advice})
    except Exception as e:
        return jsonify({"success": False, "error": f"处理失败: {str(e)}"}), 500


# ========== Function Calling  ==========
def call_deepseek_with_tools(messages, tools, system_prompt=None, temperature=0.7, max_tokens=800):
    """
    支持工具调用的 DeepSeek API 调用

    功能说明:
        发送消息到 DeepSeek API，支持 Function Calling 机制。
        AI 可以自主决定调用哪些工具来获取信息，实现多智能体协作。

    参数:
        messages: list[dict] - 消息历史列表，每个元素包含:
            - role: str - "user" 或 "assistant" 或 "tool"
            - content: str - 消息内容
        tools: list[dict] - 可用工具列表，每个工具包含:
            - type: "function"
            - function: { name, description, parameters }
        system_prompt: str | None - 系统提示词
        temperature: float - 生成温度，默认 0.7
        max_tokens: int - 最大输出 token 数，默认 800

    返回:
        dict - DeepSeek API 响应 JSON

    工具列表（定义在 /api/chat 中）:
        1. query_user_records - 查询用户最近饮食记录（小记）
        2. check_allergy_risk - 检查过敏风险（小安）
        3. get_nutrition_info - 获取菜品营养成分（小物）

    调用场景:
        /api/chat 接口使用 Function Calling 实现智能体协作
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": "auto"
    }
    if system_prompt:
        # 将 system 消息放在最前面
        messages.insert(0, {"role": "system", "content": system_prompt})
        payload["messages"] = messages

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


# 定义工具列表（工具函数实现）
def tool_query_records(user_id, days=7):
    """
    工具：小记查询并分析用户饮食记录

    参数:
        user_id: str - 用户ID
        days: int - 查询最近几天的记录，默认 7

    返回:
        str - 小记的分析结果

    用途:
        作为 Function Calling 的工具函数，供 AI 调用
    """
    return agent.call_xiaoji_for_records(user_id, days)


def tool_check_allergy(dish_name, user_allergens):
    """
    工具：小安检查过敏风险

    参数:
        dish_name: str - 菜品名称
        user_allergens: list[str] - 用户过敏原列表

    返回:
        str - 过敏风险评估结果

    用途:
        作为 Function Calling 的工具函数，供 AI 调用
    """
    return agent.call_xiaoan_for_allergy(dish_name, user_allergens)


def tool_nutrition_analysis(dish_name, user_profile):
    """
    工具：小物分析营养成分

    参数:
        dish_name: str - 菜品名称
        user_profile: dict - 用户健康档案

    返回:
        str - 小物的营养分析结果

    用途:
        作为 Function Calling 的工具函数，供 AI 调用
    """
    return agent.call_xiaowu_for_nutrition(dish_name, user_profile)


def tool_meal_plan(user_profile):
    """小盘：生成膳食推荐 """
    return agent.call_xiaopan_for_meal_plan(user_profile)


# ======= /api/chat 接口，使用 Function Calling =======
@app.route('/api/chat', methods=['POST'])
def chat():
    """
    对话接口（多智能体协作，支持 Function Calling）

    功能说明:
        这是整个系统的核心对话接口，支持多智能体协作。
        用户发送消息后，系统会：
        1. 自动获取用户最近的饮食记录作为上下文
        2. AI（小膳）分析用户意图，决定调用哪些工具
        3. 工具执行后返回结果（小记/小物/小安/小盘的分析）
        4. AI 综合所有信息，生成最终回复

    请求格式:
        POST /api/chat
        Content-Type: application/json
        {
            "message": "帮我制定一个减脂食谱",      // 用户消息
            "userProfile": {                       // 用户健康档案
                "bmi": "22.5",
                "bmiStatus": "正常",
                "activity": "中",
                "goal": "减脂",
                "allergens": ["花生"],
                "user_id": "user_1234567890_abc123"
            },
            "history": [                           // 对话历史（最近6条）
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "userId": "user_1234567890_abc123"     // 用户ID
        }

    返回格式:
        成功: {
            "success": true,
            "content": "综合回复内容",              // AI 最终回复
            "blocks": [                           // 结构化内容块
                {"type": "text", "data": "..."}
            ],
            "actions": [                          // 操作按钮
                {"label": "有帮助", "action": "helpful"},
                {"label": "换一个", "action": "retry"}
            ]
        }
        失败: {"success": false, "error": "错误信息"}

    前端调用:
        小程序 chat.js -> sendMessage() -> callAI() -> wx.request -> /api/chat

    多智能体协作流程:
        1. 用户发送消息
        2. 系统自动注入用户最近饮食记录（小记提供）
        3. AI 分析用户意图
        4. 如需更多信息，AI 调用工具:
           - query_user_records -> 小记分析记录
           - check_allergy_risk -> 小安检查过敏
           - get_nutrition_info -> 小物分析营养
           - get_meal_plan -> 小盘生成膳食推荐  ← 新增！
        5. 工具返回结果
        6. AI 综合所有信息，生成最终回复
    """

    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"success": False, "error": error_msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求数据为空"}), 400

    message = data.get('message', '')
    user_profile = data.get('userProfile', {})
    history = data.get('history', [])

    # 获取 user_id
    user_id = data.get('userId')
    if not user_id or user_id == 'default_user':
        user_id = user_profile.get('user_id', 'default_user')

    if not message:
        return jsonify({"success": False, "error": "消息内容为空"}), 400

    # ==================== 构建工具列表（4个工具） ====================
    base_tools = [
        # 工具1：小记 - 查询用户饮食记录
        {
            "type": "function",
            "function": {
                "name": "query_user_records",    # 工具名称
                # 工具描述
                "description": "查询用户最近的饮食记录。当用户询问'我最近吃了什么''帮我分析一下饮食记录''我最近吃得怎么样''我的饮食习惯如何'时调用此工具。",
                # 参数定义
                "parameters": {
                    "type": "object",    # 参数对象
                    # 具体参数属性列表
                    "properties": {
                        "days": {
                            "type": "integer",    # 参数类型(如int)
                            "description": "查询最近几天的记录，默认7"
                        }
                    },
                    # 必填参数列表
                    "required": []
                }
            }
        },

        # 工具2：小安 - 检查过敏风险
        {
            "type": "function",
            "function": {
                "name": "check_allergy_risk",
                "description": "检查特定菜品是否含有用户的过敏原。当用户询问'这道菜我能吃吗''会不会过敏''有没有xxx过敏原''这个菜安全吗'时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dish_name": {
                            "type": "string",
                            "description": "要检查的菜品名称"
                        }
                    },
                    "required": ["dish_name"]
                }
            }
        },

        # 工具3：小物 - 获取菜品营养成分
        {
            "type": "function",
            "function": {
                "name": "get_nutrition_info",
                "description": "获取菜品的详细营养成分（热量、蛋白质、碳水、脂肪）。当用户询问'这道菜热量高吗''营养怎么样''蛋白质多少''脂肪含量''含多少卡路里'时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dish_name": {
                            "type": "string",
                            "description": "要查询的菜品名称"
                        }
                    },
                    "required": ["dish_name"]
                }
            }
        },

        # 工具4：小盘 - 整体膳食推荐
        {
            "type": "function",
            "function": {
                "name": "get_meal_plan",
                "description": "根据用户的健康档案（BMI、运动量、健康目标、过敏原）生成整体的膳食推荐方案，包含早餐、午餐、晚餐的具体菜品建议。"
                               "当用户询问'我该吃什么''给我推荐食谱''帮我制定饮食计划''减肥该吃什么''增肌怎么吃''膳食建议''一日三餐怎么吃''健康饮食方案'时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    tools = base_tools

    # ===== 构建消息历史 =====
    messages = []
    for h in history[-6:]:
        if h.get('role') in ['user', 'assistant']:
            messages.append({"role": h['role'], "content": h.get('content', '')})

    # ===== 强制获取饮食记录，拼接到用户消息中 =====
    records_text = ""
    records = get_user_records(user_id, days=7, limit=10)
    if records:
        records_text = "\n\n【小记提供的用户最近饮食记录】\n"
        for r in records[:5]:
            records_text += f"- {r.get('recorded_at', '未知时间')} 吃了 {r.get('dish_name', '')}（{r.get('calories', 0)}kcal）\n"
        if len(records) > 5:
            records_text += f"... 共 {len(records)} 条记录"
    else:
        records_text = "\n\n【小记提供的用户最近饮食记录】暂无记录。"

    # 把记录拼接到当前用户消息中
    full_message = message + records_text
    messages.append({"role": "user", "content": full_message})

    # ===== 小膳系统提示词 =====
    system_prompt = """你是"小膳"，一个多智能体协作系统的总协调员。你身边有四位专家助手：

- 小物：擅长分析菜品营养成分（热量、蛋白质、碳水、脂肪）
- 小记：擅长分析用户饮食历史记录（用餐规律、营养均衡度）
- 小安：擅长检查过敏风险（识别过敏原、预警）
- 小盘：擅长整体膳食推荐（制定一日三餐食谱）

【你的工作方式】
1. 当用户询问具体菜品时，调用小物（营养）和小安（过敏）
2. 当用户询问饮食习惯时，调用小记（记录分析）
3. 当用户询问吃什么、怎么吃时，调用小盘（膳食推荐）
4. 可以同时调用多个工具，综合各位专家的意见

【回答要求】
1. 引用专家分析时，必须明确说出"小物说""小记发现""小安提醒""小盘建议"等
2. 严禁使用任何 emoji 表情符号（如❌✅⚠️📅等）
3. 可以采用分点的方式组织回答，让用户更清晰
4. 语言要自然亲切，像朋友聊天一样，不要生硬
5. 综合各位专家的意见，给出最终的、有优先级的建议
6. 【较为重要】如果有过敏源，小安的回答要放前面，做到优先提醒
7. 尽量控制在600字以内
8. 纯文本输出，不使用任何 markdown 格式

"""
    # 注入用户健康档案信息
    context_info = f"用户健康档案：BMI {user_profile.get('bmi', '--')}，运动量 {user_profile.get('activity', '--')}，目标 {user_profile.get('goal', '--')}"
    if user_profile.get('allergens'):
        context_info += f"，过敏原：{', '.join(user_profile['allergens'])}"
    system_prompt += "\n" + context_info

    # ===== 调用 DeepSeek =====
    try:
        # ===== 第1次调用：让 AI 分析用户意图，决定是否调用工具 =====
        # 发送用户消息 + 4个工具定义，AI 返回是否要调用工具
        response = call_deepseek_with_tools(messages, tools, system_prompt=system_prompt)

        # 获取 AI 返回的工具调用指令（如果有）
        tool_calls = response['choices'][0]['message'].get('tool_calls')

        # ===== 判断：AI 是否决定调用工具 =====
        if tool_calls:
            # ---- 如果AI 决定调用工具 ----
            tool_results = []

            # 遍历 AI 要调用的每个工具
            for tc in tool_calls:
                # 获取工具名称（如 "query_user_records"）
                func_name = tc['function']['name']
                # 解析工具参数（如 {"days": 7}）
                args = json.loads(tc['function']['arguments'])

                # ----- 根据工具名称，执行对应的后端函数 -----
                if func_name == 'query_user_records':
                    # 小记：查询饮食记录
                    days = args.get('days', 7)
                    result = tool_query_records(user_id, days)

                elif func_name == 'check_allergy_risk':
                    # 小安：检查过敏风险
                    dish_name = args.get('dish_name')
                    result = tool_check_allergy(dish_name, user_profile.get('allergens', []))

                elif func_name == 'get_nutrition_info':
                    # 小物：获取营养成分
                    dish_name = args.get('dish_name')
                    profile_with_user = user_profile.copy()
                    profile_with_user['user_id'] = user_id
                    result = tool_nutrition_analysis(dish_name, profile_with_user)

                elif func_name == 'get_meal_plan':
                    # 小盘：生成膳食推荐（新增）
                    profile_with_user = user_profile.copy()
                    profile_with_user['user_id'] = user_id
                    result = tool_meal_plan(profile_with_user)

                else:
                    # 未知工具（理论上不会发生）
                    result = "未知工具"

                # 收集工具执行结果（格式要求：tool_call_id + role + content）
                tool_results.append({
                    "tool_call_id": tc['id'],  # 对应哪次调用
                    "role": "tool",    # 固定值，表示这是工具返回
                    "content": result  # 工具执行的结果
                })

            # 把 AI 的"工具调用指令"加入消息历史
            messages.append(response['choices'][0]['message'])
            # 把工具执行结果加入消息历史
            messages.extend(tool_results)

            # ===== 第2次调用：让 AI 综合所有工具结果，生成最终回复 =====
            # 注意：这次 tools=[]，不再给 AI 工具，只让它综合信息
            final_response = call_deepseek_with_tools(messages, [], system_prompt=system_prompt)
            final_content = final_response['choices'][0]['message']['content']
        else:
            # ---- AI 决定不调用工具，直接回答 ----
            final_content = response['choices'][0]['message']['content']

        final_content = agent.clean_response(final_content)

        return jsonify({
            'success': True,
            'content': final_content,
            'blocks': [{'type': 'text', 'data': final_content}],
            'actions': [
                {'label': '有帮助', 'action': 'helpful'},
                {'label': '换一个', 'action': 'retry'}
            ]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
