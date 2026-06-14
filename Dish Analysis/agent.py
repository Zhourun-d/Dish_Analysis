"""
AI 提示词构建模块
负责构建发送给 DeepSeek 的提示词，以及清理 AI 返回的内容
"""

import re


def clean_response(text):
    """
    清理 AI 返回的文本，移除 markdown 格式符号

    功能:
        - 移除 ** 加粗标记
        - 移除 # 标题标记
        - 移除其他 markdown 符号（* # ` >）
        - 将连续多个换行压缩为两个换行

    参数:
        text: AI 返回的原始文本

    返回:
        清理后的纯文本
    """
    # 移除 markdown 加粗标记 **
    text = re.sub(r'\*\*', '', text)

    # 移除 markdown 标题标记（行首的 # 号）
    # ^#+ 匹配行首的一个或多个 #，\s* 匹配后面的空白字符
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # 移除其他可能残留的 markdown 符号
    # * # ` > 这些符号在 markdown 中有特殊含义
    text = re.sub(r'[*#`>]', '', text)

    # 将连续 3 个或以上的换行压缩为 2 个换行
    # 避免输出中出现过多空白行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除首尾空白字符并返回
    return text.strip()


def build_xiaopan_prompt(bmi, bmi_status, activity, goal, allergens):
    """
    构建给 DeepSeek 的提示词 - 小盘（整体膳食推荐）

    参数:
        bmi: 用户的 BMI 值（如 22.5）
        bmi_status: BMI 状态（偏瘦/正常/偏胖/肥胖）
        activity: 运动量（低/中/高）
        goal: 健康目标（减脂/增肌/均衡）
        allergens: 过敏原列表，如 ['花生', '海鲜']

    返回:
        完整的提示词字符串
    """
    # 处理过敏原：如果有过敏原则用顿号连接，否则显示"无"
    allergen_text = "、".join(allergens) if allergens else "无"

    prompt = f"""你是一位专业的膳食推荐师，名字叫"小盘"。请根据以下用户信息，给出简洁、实用的膳食推荐建议。

用户信息：
- BMI：{bmi}（{bmi_status}）
- 运动量：{activity}
- 健康目标：{goal}
- 过敏原：{allergen_text}

要求：
1. 用第一人称"小盘"的口吻，亲切自然地说话
2. 先简要分析用户身体状况，再给出具体膳食建议
3. 建议要具体，包含早餐、午餐、晚餐的推荐
4. 字数控制在300字以内，语言简洁
5. 不要有任何markdown格式（如**加粗**、#标题等），纯文本输出
6. 不要分点罗列，用自然段落表达
7. 如果有过敏原，提醒用户注意避开相关食材

请直接输出推荐内容："""
    return prompt


def build_xiaowu_prompt(dish, user_profile):
    """
    构建给 DeepSeek 的提示词 - 小物（针对特定菜品的个性化分析）

    参数:
        dish: 菜品信息字典，包含:
            - name: 菜品名称
            - calories: 热量（千卡/100g）
            - protein: 蛋白质（g/100g）
            - carbs: 碳水化合物（g/100g）
            - fat: 脂肪（g/100g）
            - allergens: 过敏原信息
        user_profile: 用户健康档案字典，包含:
            - bmi: BMI 值
            - bmiStatus: BMI 状态
            - activity: 运动量
            - goal: 健康目标
            - allergens: 过敏原列表

    返回:
        完整的提示词字符串
    """
    # 从菜品字典中提取信息，如果字段缺失则使用默认值
    dish_name = dish.get('name', '该菜品')
    calories = dish.get('calories', '未知')
    protein = dish.get('protein', '未知')
    carbs = dish.get('carbs', '未知')
    fat = dish.get('fat', '未知')
    dish_allergens = dish.get('allergens', '无')

    # 从用户档案中提取信息
    bmi = user_profile.get('bmi', '--')
    bmi_status = user_profile.get('bmiStatus', '')
    activity = user_profile.get('activity', '--')
    goal = user_profile.get('goal', '--')
    user_allergens = user_profile.get('allergens', [])
    user_allergen_text = "、".join(user_allergens) if user_allergens else "无"

    prompt = f"""你是"小物"，一位专业且亲切的营养分析智能体。请根据用户识别出的菜品和用户的健康状况，给出简洁、个性化的饮食建议。

【菜品信息】
名称：{dish_name}
每100克营养成分：热量{calories}千卡，蛋白质{protein}克，碳水化合物{carbs}克，脂肪{fat}克。
菜品可能含有的过敏原：{dish_allergens}

【用户健康档案】
BMI：{bmi}（{bmi_status}）
运动量：{activity}
健康目标：{goal}
用户已知过敏原：{user_allergen_text}

要求：
1. 以"小物"的第一人称口吻，亲切自然，语气温和。
2. 先简单评价这道菜对该用户是否合适（考虑热量、营养素、过敏风险）。
3. 结合用户的健康目标（减脂/增肌/均衡），给出具体的食用建议（例如建议食用量、搭配什么一起吃、烹饪改进等）。
4. 如果菜品含有用户过敏原，必须明确提醒避开，并建议替换菜品。
5. 总字数控制在200字以内，纯文本，不使用任何 markdown 格式和表情符号。
6. 不要罗列条目，用连贯的一段或两段话表达。

请直接输出建议内容："""
    return prompt
