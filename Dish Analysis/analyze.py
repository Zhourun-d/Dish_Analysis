"""
菜品识别与膳食推荐服务
- 菜品识别：使用 YOLO 模型识别图片中的菜品
- 小盘推荐：根据用户健康数据生成整体膳食建议
- 小物分析：针对特定菜品给出个性化营养建议
"""

import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from ultralytics import YOLO
import requests
from dotenv import load_dotenv

from nutrition_lib import get_nutrition, get_all_dishes
import agent

# 应用初始化
app = Flask(__name__)
CORS(app)  # 允许跨域请求，方便前端调用

# 加载 .env 文件中的环境变量（存储 API Key 等敏感信息）
load_dotenv()

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 检查 API Key 是否配置，未配置时给出警告
if not DEEPSEEK_API_KEY:
    print("警告: DEEPSEEK_API_KEY 环境变量未设置，请检查 .env 文件")
    print("提示: 在项目根目录创建 .env 文件，添加 DEEPSEEK_API_KEY=你的密钥")

# 加载 YOLO 菜品识别模型（全局加载一次，避免每次请求都重新加载）
model = YOLO("./model/best.pt")

# 加载类别 ID 到菜品名称的映射表
# Excel 文件第一列是类别 ID，第二列是对应的菜品名称
df = pd.read_excel('./dish_mapping/class_names.xlsx', header=None)
id_to_name = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # 转换为字典，便于快速查找


def calibrated_confidence(probs):
    raw_conf = probs.top1conf.item()
    # 分段线性映射
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


def check_api_key():
    """
    检查 DeepSeek API Key 是否配置有效
    """
    if not DEEPSEEK_API_KEY:
        return False, "DeepSeek API 密钥未配置"
    return True, None


# ==================== API 路由 ====================
@app.route('/recognize', methods=['POST'])
def recognize():
    """
    菜品识别接口
    接收前端上传的图片，使用 YOLO 模型识别菜品，返回识别结果和营养信息

    请求格式:
        POST /recognize
        form-data: image=图片文件

    返回格式:
        成功: {
            "success": true,
            "dish": {
                "name": "菜品名称",
                "probability": 置信度百分比,
                "calories": 热量(千卡/100g),
                "protein": 蛋白质(g/100g),
                "carbs": 碳水化合物(g/100g),
                "fat": 脂肪(g/100g),
                "allergens": "过敏原信息",
                "tips": "食用建议"
            }
        }
        失败: {"success": false, "error": "错误信息"}
    """
    # 1. 校验请求中是否包含图片
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '未上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '图片文件名为空'}), 400

    # 2. 保存临时文件
    # NamedTemporaryFile: 创建临时文件，delete=False 表示关闭后不自动删除
    # suffix='.jpg': 指定文件后缀，让模型能正确识别格式
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)  # 将上传的图片保存到临时文件
    temp_file.close()  # 关闭文件句柄，释放资源

    try:
        # 3. 调用 YOLO 模型进行识别
        results = model(temp_file.name)
        probs = results[0].probs  # 获取概率分布（分类任务的输出）

        # 4. 获取预测结果
        pred_id = probs.top1  # 概率最高的类别 ID
        dish_name = id_to_name[pred_id]  # 根据 ID 查找菜品名称

        # 5. 计算校准后的置信度，并转为百分比
        raw_conf = calibrated_confidence(probs)
        confidence_percent = int(raw_conf * 100)

        # 6. 获取该菜品的营养数据
        nutrition = get_nutrition(dish_name)

        if nutrition:
            # 构建完整的菜品信息
            dish_info = {
                'name': dish_name,
                'probability': confidence_percent,
                'calories': nutrition['calories'],
                'protein': nutrition['protein'],
                'carbs': nutrition['carbs'],
                'fat': nutrition['fat'],
                'allergens': nutrition['allergens'],
                'tips': nutrition['tips']
            }
        else:
            # 营养数据不存在时的降级处理
            dish_info = {
                'name': dish_name,
                'probability': confidence_percent,
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'allergens': '未知',
                'tips': f'暂未收录 {dish_name} 的营养数据，当前已收录: {", ".join(get_all_dishes())}'
            }

        return jsonify({'success': True, 'dish': dish_info})

    except Exception as e:
        # 识别过程中发生异常，返回错误信息
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        # 无论成功还是失败，都要删除临时文件，避免磁盘空间浪费
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.route('/api/recommend', methods=['POST'])
def xiaopan_advice():
    """
    小盘膳食推荐接口
    根据用户的健康数据（BMI、运动量、目标、过敏原）生成整体膳食建议

    请求格式:
        POST /api/recommend
        JSON: {
            "bmi": 22.5,
            "bmiStatus": "正常",
            "activity": "中等",
            "goal": "减脂",
            "allergens": ["花生", "海鲜"]
        }

    返回格式:
        成功: {"success": true, "recommendation": "建议内容"}
        失败: {"error": "错误信息"}
    """
    # 1. 检查 API Key
    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"error": error_msg}), 500

    # 2. 获取请求数据
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求数据为空"}), 400

    # 3. 提取用户信息
    bmi = data.get('bmi', '--')
    bmi_status = data.get('bmiStatus', '')
    activity = data.get('activity', '--')
    goal = data.get('goal', '--')
    allergens = data.get('allergens', [])

    # 4. 调用 agent 模块构建提示词
    prompt = agent.build_xiaopan_prompt(bmi, bmi_status, activity, goal, allergens)

    # 5. 调用 DeepSeek API
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
        "temperature": 0.7,  # 控制回答的随机性，0.7 较为平衡
        "max_tokens": 500    # 限制回答长度
    }

    try:
        # 发送请求到 DeepSeek，超时时间 30 秒
        response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # 如果状态码不是 2xx，抛出异常

        result = response.json()
        recommendation = result['choices'][0]['message']['content']

        # 清理可能残留的 markdown 格式符号
        recommendation = agent.clean_response(recommendation)

        return jsonify({
            "success": True,
            "recommendation": recommendation
        })

    except requests.exceptions.Timeout:
        return jsonify({"error": "请求超时，请稍后重试"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"API 请求失败: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


@app.route('/api/xiaowu_advice', methods=['POST'])
def xiaowu_advice():
    """
    小物营养分析接口
    根据菜品信息和用户健康档案，给出针对性的营养建议

    请求格式:
        POST /api/xiaowu_advice
        JSON: {
            "dish": {
                "name": "菜品名称",
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

    返回格式:
        成功: {"success": true, "advice": "分析建议"}
        失败: {"success": false, "error": "错误信息"}
    """
    # 1. 检查 API Key
    valid, error_msg = check_api_key()
    if not valid:
        return jsonify({"success": False, "error": error_msg}), 500

    # 2. 获取请求数据
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求数据为空"}), 400

    dish = data.get('dish')
    user_profile = data.get('userProfile')

    # 3. 校验必要字段
    if not dish or not user_profile:
        return jsonify({"success": False, "error": "缺少菜品或用户信息"}), 400

    # 4. 调用 agent 模块构建提示词
    prompt = agent.build_xiaowu_prompt(dish, user_profile)

    # 5. 调用 DeepSeek API
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

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "请求超时，请稍后重试"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "error": f"API 请求失败: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"处理失败: {str(e)}"}), 500


# ==================== 启动 ====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
