"""
AI 提示词构建模块
负责构建发送给 DeepSeek 的提示词，以及清理 AI 返回的内容

模块功能概述:
    1. clean_response: 清理 AI 返回文本中的 Markdown 格式符号
    2. build_xiaopan_prompt: 构建小盘（膳食推荐师）的提示词
    3. build_xiaowu_prompt: 构建小物（营养分析）的提示词
    4. build_xiaoji_prompt: 构建小记（记录分析）的提示词
    5. build_xiaoan_prompt: 构建小安（过敏预警）的提示词
    6. call_xiaoji_for_records: 调用 小记分析饮食记录（含降级逻辑）
    7. call_xiaoan_for_allergy: 调用小安检查过敏风险（含降级逻辑）
    8. call_xiaowu_for_nutrition: 调用小物分析菜品营养（含降级逻辑）
    9. call_xiaopan_for_meal_plan: 调用小盘生成膳食推荐（含降级逻辑）

前端调用链路:
    小程序 analyse 页面 -> /api/xiaopan_advice -> build_xiaopan_prompt -> DeepSeek
    小程序 index 页面 -> /api/xiaowu_advice -> build_xiaowu_prompt -> DeepSeek
    小程序 record 页面 -> /api/xiaoji_advice -> build_xiaoji_prompt -> DeepSeek
    小程序 chat 页面 -> /api/chat -> 各 build 函数 -> DeepSeek (Function Calling)
"""

import re
from services import get_user_records, call_deepseek
from nutrition_lib import get_nutrition


def clean_response(text):
    """
    清理 AI 返回的文本，移除 markdown 格式符号

    """
    if not text:
        return ""
    # 移除 markdown 加粗标记 **
    text = re.sub(r'\*\*', '', text)

    # 移除 markdown 标题标记（行首的 # 号）
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # 移除其他可能残留的 markdown 符号
    text = re.sub(r'[*#`>]', '', text)

    # 将连续 3 个或以上的换行压缩为 2 个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除首尾空白字符并返回
    return text.strip()


def build_xiaopan_prompt(bmi, bmi_status, activity, goal, allergens, records=None):
    """
    构建给 DeepSeek 的提示词 - 小盘（整体膳食推荐）

    小盘角色: 专业的膳食推荐师，根据用户健康档案和历史饮食记录，
             给出整体性的膳食推荐建议（早餐、午餐、晚餐）。

    参数:
        bmi: float - 用户的 BMI 值，如 22.5
        bmi_status: str - BMI 状态描述，如 "正常"、"偏瘦"、"超重"、"肥胖"
        activity: str - 运动量，如 "很少运动"、"每周1-3次"、"每周4-6次"、"每天运动"
        goal: str - 健康目标，如 "减脂"、"增肌"、"均衡营养"
        allergens: list[str] - 用户已知过敏原列表，如 ["花生", "虾"]
        records: list[dict] | None - 用户最近饮食记录列表，每个元素包含:
            - recorded_at: str - 记录时间（格式: "YYYY-MM-DD HH:MM"）
            - dish_name: str - 菜品名称
            - calories: float - 热量（千卡）

    返回:
        str - 完整的提示词字符串，可直接发送给 DeepSeek API

    提示词要求:
        1. 第一人称 "小盘" 口吻
        2. 包含早餐、午餐、晚餐推荐
        3. 结合历史记录避免重复推荐
        4. 提醒过敏原
        5. 300字以内，纯文本，无 Markdown
        6. 引用小记（记录分析）和小安（过敏预警）

    前端调用链路:
        小程序 analyse.js -> startAnalyse() ->
        POST /api/xiaopan_advice ->
        build_xiaopan_prompt() ->
        DeepSeek API ->
        返回 recommendation 字段

    请求格式 (来自前端 analyse.js):
        {
            "bmi": "22.5",
            "bmiStatus": "正常",
            "activity": "中",
            "goal": "减脂",
            "allergens": ["花生", "海鲜"],
            "userId": "user_1234567890_abc123"
        }

    返回格式 (给前端):
        {
            "success": true,
            "recommendation": "小盘说：根据您的BMI和减脂目标..."
        }
    """
    allergen_text = "、".join(allergens) if allergens else "无"

    # 格式化历史记录
    records_text = ""
    if records:
        records_text = "\n【用户最近饮食记录】\n"
        for r in records[:5]:  # 最多取5条
            records_text += f"- {r['recorded_at']} 吃了 {r['dish_name']}（{r['calories']}kcal）\n"
        if len(records) > 5:
            records_text += f"... 共 {len(records)} 条记录\n"
    else:
        records_text = "\n【用户最近饮食记录】暂无记录。\n"

    prompt = f"""你是一位专业的膳食推荐师，名字叫"小盘"。请根据以下用户信息，给出简洁、实用的膳食推荐建议。

用户信息：
- BMI：{bmi}（{bmi_status}）
- 运动量：{activity}
- 健康目标：{goal}
- 过敏原：{allergen_text}
{records_text}

要求：
1. 用第一人称"小盘"的口吻，亲切自然地说话
2. 先简要分析用户身体状况，再给出具体膳食建议
3. 建议要具体，包含早餐、午餐、晚餐的推荐
4. 结合用户的历史饮食记录，避免重复推荐近期已吃过的菜品，并提醒营养均衡
5. 如果有过敏原，提醒用户注意避开相关食材
6. 字数控制在300字以内，语言简洁
7. 不要有任何markdown格式（如**加粗**、#标题等），纯文本输出
8. 不要分点罗列，用自然段落表达

9. 【协作要求】如果参考了用户的历史饮食记录，请以"我的朋友小记看到您最近..."或"我的搭档小记发现您..."等方式引用，体现团队协作。
10. 【协作要求】如果菜品含有用户过敏原，请以"我的朋友小安提醒您..."或者"我的搭档小安发现您..."等方式引用。
11. 可以适当换行。

请直接输出推荐内容："""
    return prompt


def build_xiaowu_prompt(dish, user_profile):
    """
    构建给 DeepSeek 的提示词 - 小物（针对特定菜品的个性化分析）

    小物角色: 营养分析智能体，针对用户识别出的特定菜品，
             结合用户健康状况给出个性化营养建议。

    参数:
        dish: dict - 菜品信息，包含:
            - name: str - 菜品名称
            - calories: float - 热量（千卡/100g）
            - protein: float - 蛋白质（g/100g）
            - carbs: float - 碳水化合物（g/100g）
            - fat: float - 脂肪（g/100g）
            - allergens: str - 菜品可能含有的过敏原，如 "花生, 鸡蛋" 或 "无"
        user_profile: dict - 用户健康档案，包含:
            - bmi: float - BMI值
            - bmiStatus: str - BMI状态
            - activity: str - 运动量
            - goal: str - 健康目标
            - allergens: list[str] - 用户过敏原列表
            - recent_records: list[dict] - 最近饮食记录（可选）

    返回:
        str - 完整的提示词字符串，可直接发送给 DeepSeek API

    提示词要求:
        1. 第一人称 "小物" 口吻
        2. 评价菜品对用户的合适程度
        3. 结合健康目标和历史记录给出具体建议
        4. 过敏提醒
        5. 200字以内，纯文本
        6. 引用小记（记录分析）和小安（过敏预警）

    前端调用链路:
        小程序 index.js -> startXiaowuAnalyse() ->
        POST /api/xiaowu_advice ->
        build_xiaowu_prompt() ->
        DeepSeek API ->
        返回 advice 字段

    请求格式 (来自前端 index.js):
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
            "userId": "user_1234567890_abc123"
        }

    返回格式 (给前端):
        {
            "success": true,
            "advice": "小物说：宫保鸡丁含有花生，与您的过敏原冲突..."
        }
    """
    dish_name = dish.get('name', '该菜品')
    calories = dish.get('calories', '未知')
    protein = dish.get('protein', '未知')
    carbs = dish.get('carbs', '未知')
    fat = dish.get('fat', '未知')
    dish_allergens = dish.get('allergens', '无')

    bmi = user_profile.get('bmi', '--')
    bmi_status = user_profile.get('bmiStatus', '')
    activity = user_profile.get('activity', '--')
    goal = user_profile.get('goal', '--')
    user_allergens = user_profile.get('allergens', [])
    user_allergen_text = "、".join(user_allergens) if user_allergens else "无"

    # 获取历史记录
    records = user_profile.get('recent_records', [])
    records_text = ""
    if records:
        records_text = "\n【用户最近饮食记录】\n"
        for r in records[:5]:
            records_text += f"- {r['recorded_at']} 吃了 {r['dish_name']}（{r['calories']}kcal）\n"
        if len(records) > 5:
            records_text += f"... 共 {len(records)} 条记录\n"
    else:
        records_text = "\n【用户最近饮食记录】暂无记录。\n"

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
{records_text}

要求：
1. 以"小物"的第一人称口吻，亲切自然，语气温和。
2. 先简单评价这道菜对该用户是否合适（考虑热量、营养素、过敏风险）。
3. 结合用户的健康目标（减脂/增肌/均衡）和最近饮食记录，给出具体的食用建议（例如建议食用量、搭配什么一起吃、烹饪改进等）。
4. 如果菜品含有用户过敏原，必须明确提醒避开，并建议替换菜品。
5. 总字数控制在200字以内，纯文本，不使用任何 markdown 格式和表情符号。
6. 不要罗列条目，用连贯的一段或两段话表达。

7. 【协作要求】如果参考了用户的历史饮食记录，请以"我的朋友小记看到您最近..."或"我的搭档小记发现您..."的方式引用，体现团队协作。
8. 【协作要求】如果菜品含有用户过敏原，请以"我的朋友小安提醒您..."的方式引用。
9. 可以适当换行，比如遇到转折点（有过敏源）了，可以换行。

请直接输出建议内容："""
    return prompt


def build_xiaoji_prompt(records, user_profile):
    """
    构建给 DeepSeek 的提示词 - 小记（饮食记录分析）

    小记角色: 饮食记录分析师，分析用户的历史饮食记录，
             关注用餐时间规律、营养均衡度、热量控制等。

    参数:
        records: list[dict] - 用户饮食记录列表，每个元素包含:
            - recorded_at: str - 记录时间（如 "2026-06-19 12:30"）
            - dish_name: str - 菜品名称
            - calories: float - 热量
            - protein: float - 蛋白质（可选）
        user_profile: dict - 用户健康档案，包含:
            - bmi: float - BMI值
            - bmiStatus: str - BMI状态
            - activity: str - 运动量
            - goal: str - 健康目标
            - allergens: list[str] - 用户过敏原列表

    返回:
        str - 完整的提示词字符串，可直接发送给 DeepSeek API

    前端调用链路:
        小程序 record.js -> startXiaojiAnalyse() ->
        POST /api/xiaoji_advice ->
        build_xiaoji_prompt() ->
        DeepSeek API ->
        返回 advice 字段

    请求格式 (来自前端 record.js):
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

    返回格式 (给前端):
        {
            "success": true,
            "advice": "小记说：小记注意到您最近三餐时间比较规律..."
        }
    """
    # ===== 记录格式化时带上时间 =====
    dish_names_with_time = []
    for r in records:
        recorded_at = r.get('recorded_at', '未知时间')
        dish_name = r.get('dish_name', '')
        calories = r.get('calories', 0)
        # 格式化时间：如果包含日期时间则直接显示，否则显示原始值
        dish_names_with_time.append(f"【{recorded_at}】{dish_name}（{calories}kcal）")

    dish_names_text = "\n".join(dish_names_with_time) if dish_names_with_time else "暂无记录"

    total_calories = sum(r.get('calories', 0) for r in records)
    avg_calories = total_calories / len(records) if records else 0
    total_protein = sum(r.get('protein', 0) for r in records)
    avg_protein = total_protein / len(records) if records else 0

    bmi = user_profile.get('bmi', '--')
    bmi_status = user_profile.get('bmiStatus', '')
    activity = user_profile.get('activity', '--')
    goal = user_profile.get('goal', '--')
    allergens = user_profile.get('allergens', [])
    allergen_text = "、".join(allergens) if allergens else "无"

    prompt = f"""你是"小记"，一位专业且亲切的饮食记录分析师。请根据用户的历史饮食记录和健康状况，给出总结性建议。

【用户饮食记录】（按时间排序，包含具体用餐时间）
共 {len(records)} 条记录
{dish_names_text}

【统计数据】
平均热量：{avg_calories:.1f} 千卡/份
平均蛋白质：{avg_protein:.1f} 克/份

【用户健康档案】
BMI：{bmi}（{bmi_status}）
运动量：{activity}
健康目标：{goal}
用户已知过敏原：{allergen_text}

【分析要求 - 重要】
1. 请特别关注用户的用餐时间规律：
   - 是否有固定的三餐时间？
   - 两餐间隔是否合理（4-6小时为佳）？
   - 有没有深夜进食或长时间未进食的情况？
   - 用餐时间是否与健康目标匹配（如减脂者不宜太晚进食）？
   - 【同样重要】如果没有固定三餐或者时间不合理，可以适当问一问是不是没提供，或者用餐的时间信息提供不完整，防止误会

2. 以"小记"的第一人称口吻说话，可以用"小记注意到..."或"小记发现..."等方式表达。
3. 分析用户的饮食习惯（如菜品多样性、营养均衡度、热量控制情况）。
4. 结合用户的健康目标（减脂/增肌/均衡），给出具体的改进建议。
5. 如果用户有过敏原，检查记录中是否有涉及过敏的菜品并提醒。
6. 总字数控制在200字以内，纯文本，不使用任何 markdown 格式和表情符号。
7. 不要罗列条目，用连贯的一段话表达。

请直接输出分析内容："""
    return prompt


#
def build_xiaoan_prompt(dish_name, dish_allergens, user_allergens):
    """
    构建给 DeepSeek 的提示词 - 小安（过敏风险预警）

    小安角色: 过敏风险预警师，精准识别菜品中的过敏原，
             保障用户的饮食安全。

    参数:
        dish_name: str - 菜品名称
        dish_allergens: str - 菜品可能含有的过敏原，如 "花生, 鸡蛋" 或 "无"
        user_allergens: list[str] - 用户已知过敏原列表，如 ["花生", "虾"]

    返回:
        str - 完整的提示词字符串，可直接发送给 DeepSeek API

    提示词要求:
        1. 第一人称 "小安" 口吻，语气果断清晰
        2. 判断是否有过敏风险
        3. 如有风险，明确指出过敏原并给出警告
        4. 推荐1-2个安全的替代菜品
        5. 150字以内，纯文本

    注意:
        此函数目前仅在 agent.py 内部定义，但实际过敏检查逻辑
        在 call_xiaoan_for_allergy 中直接实现（不调用 DeepSeek），
        以提升响应速度和可靠性。
    """
    user_allergen_text = "、".join(user_allergens) if user_allergens else "无"

    prompt = f"""你是"小安"，一位专业的过敏风险预警师。你的职责是精准识别菜品中的过敏原，保障用户的饮食安全。

【菜品信息】
名称：{dish_name}
菜品可能含有的过敏原：{dish_allergens}

【用户信息】
用户已知过敏原：{user_allergen_text}

请完成以下分析：
1. 判断该菜品是否含有用户已知的过敏原
2. 如果有过敏风险，明确指出是哪种过敏原，并给出明确的警告
3. 推荐1-2个安全的替代菜品（避免含有用户过敏原的菜品）
4. 如果没有过敏风险，告知用户放心食用

要求：
1. 以"小安"的第一人称口吻，语气果断、清晰
2. 过敏警告要醒目，使用"警告"、"注意"等词汇
3. 回答要简洁，控制在150字以内
4. 纯文本，不使用markdown格式，不使用表情符号

请直接输出预警内容："""
    return prompt


# ===== 各智能体独立分析函数 =====
def call_xiaoji_for_records(user_id, days=7):
    """
    调用 DeepSeek 让小记分析用户的饮食记录

    功能说明:
        获取用户最近 N 天的饮食记录，构建提示词并调用 DeepSeek API，
        返回小记的分析结果。若 API 调用失败，降级返回原始记录摘要。

    参数:
        user_id: str - 用户唯一标识（从小程序缓存中的 userId 获取）
        days: int - 查询最近几天的记录，默认 7 天

    返回:
        str - 小记的分析结果，格式为 "小记说：..."

    降级返回格式:
        "小记说：您最近的饮食记录如下：\n【时间】菜品（热量kcal）"

    示例:
        >>> call_xiaoji_for_records("user_123", days=7)
        "小记说：小记注意到您最近三餐时间比较规律..."

    调用场景:
        在 /api/chat 接口中，作为 Function Calling 的工具函数被调用
    """
    records = get_user_records(user_id, days=days, limit=10)
    if not records:
        return "小记说：您最近没有饮食记录，建议开始记录以便我为您分析饮食规律。"

    # 构建小记专用提示词
    user_profile = {}  # 从全局获取，这里简化
    prompt = build_xiaoji_prompt(records, user_profile)

    # 调用 DeepSeek 获取小记的分析
    try:
        response = call_deepseek(prompt, "你是小记，一位专业的饮食记录分析师。")
        return f"小记说：{response}"
    except Exception as e:
        # 降级：返回原始数据
        summary = []
        for r in records[:5]:
            summary.append(f"【{r.get('recorded_at', '未知时间')}】{r.get('dish_name')}（{r.get('calories', 0)}kcal）")
        return f"小记说：您最近的饮食记录如下：\n" + "\n".join(summary)


def call_xiaoan_for_allergy(dish_name, user_allergens):
    """
    调用小安检查过敏风险（含降级逻辑）

    功能说明:
        从营养库获取菜品的过敏原信息，与用户过敏原进行比对，
        返回过敏风险评估结果。此函数直接进行逻辑判断，不调用 DeepSeek API。
        采用本地判断而非 AI 调用，以保证响应速度（< 10ms）。

    参数:
        dish_name: str - 菜品名称
        user_allergens: list[str] - 用户已知过敏原列表

    返回:
        str - 过敏风险评估结果，格式为 "小安说：..."

    返回场景:
        1. 菜品未收录: "小安说：暂未找到 {dish_name} 的过敏原信息..."
        2. 菜品无过敏原: "小安说：{dish_name} 不含常见过敏原..."
        3. 有过敏风险: "小安说：⚠️ 警告！{dish_name} 含有您过敏的食材..."
        4. 无过敏风险: "小安说：{dish_name} 含有 {allergens}，但未检测到您的过敏原..."
        5. 用户未设置过敏原: "小安说：{dish_name} 含有 {allergens}，您暂未设置过敏原..."

    示例:
        >>> call_xiaoan_for_allergy("宫保鸡丁", ["花生"])
        "小安说：⚠️ 警告！宫保鸡丁含有您过敏的食材：花生..."

    调用场景:
        在 /api/chat 接口中，作为 Function Calling 的工具函数被调用
        在小程序 index 页面识别菜品后，也可主动调用
    """
    nutrition = get_nutrition(dish_name)
    if not nutrition:
        return f"小安说：暂未找到 {dish_name} 的过敏原信息，建议您自行查看配料表。"

    dish_allergens = nutrition.get('allergens', '无')
    if dish_allergens == '无':
        return f"小安说：{dish_name} 不含常见过敏原，您可以放心食用。"

    if user_allergens:
        risky = [a for a in user_allergens if a in dish_allergens]
        if risky:
            return f"小安说：⚠️ 警告！{dish_name} 含有您过敏的食材：{', '.join(risky)}。建议立即避开，换成不含这些食材的菜品。"
        else:
            return f"小安说：{dish_name} 含有 {dish_allergens}，但未检测到您的过敏原。不过如果您对 {dish_allergens} 有轻微敏感，也建议少量尝试。"
    else:
        return f"小安说：{dish_name} 含有 {dish_allergens}，您暂未设置过敏原，建议根据自身情况判断。"


def call_xiaowu_for_nutrition(dish_name, user_profile):
    """
    调用小物分析菜品的营养成分（含降级逻辑）

    功能说明:
        获取菜品的营养数据，构建提示词并调用 DeepSeek API，
        返回小物的个性化营养分析建议。若 API 调用失败，降级返回基本营养数据。

    参数:
        dish_name: str - 菜品名称
        user_profile: dict - 用户健康档案，包含:
            - bmi: float - BMI值
            - bmiStatus: str - BMI状态
            - activity: str - 运动量
            - goal: str - 健康目标
            - allergens: list[str] - 用户过敏原列表
            - user_id: str - 用户ID（可选，用于获取历史记录）

    返回:
        str - 小物的分析结果，格式为 "小物说：..."

    降级返回格式:
        "小物说：{dish_name} 每100克含热量{X}kcal，蛋白质{Y}g..."

    示例:
        >>> call_xiaowu_for_nutrition("宫保鸡丁", user_profile)
        "小物说：宫保鸡丁每100克含160kcal，蛋白质15g..."

    调用场景:
        在 /api/chat 接口中，作为 Function Calling 的工具函数被调用
        在小程序 index 页面识别菜品后，也可主动调用
    """
    nutrition = get_nutrition(dish_name)
    if not nutrition:
        return f"小物说：暂未收录 {dish_name} 的营养数据，无法为您分析。"

    # 构建小物专用提示词（简化版）
    dish = {
        'name': dish_name,
        'calories': nutrition['calories'],
        'protein': nutrition['protein'],
        'carbs': nutrition['carbs'],
        'fat': nutrition['fat'],
        'allergens': nutrition['allergens']
    }
    prompt = build_xiaowu_prompt(dish, user_profile)

    try:
        response = call_deepseek(prompt, "你是小物，一位专业的营养分析智能体。")
        return f"小物说：{response}"
    except Exception as e:
        # 降级：返回基本营养数据
        return f"小物说：{dish_name} 每100克含热量{nutrition['calories']}kcal，蛋白质{nutrition['protein']}g，碳水{nutrition['carbs']}g，脂肪{nutrition['fat']}g。"


def call_xiaopan_for_meal_plan(user_profile):
    """
    调用小盘生成膳食推荐（含降级逻辑）

    功能说明:
        根据用户健康档案和历史饮食记录，构建提示词并调用 DeepSeek API，
        返回小盘的整体膳食推荐方案。若 API 调用失败，降级返回通用建议。

    参数:
        user_profile: dict - 用户健康档案，包含:
            - bmi: float - BMI值
            - bmiStatus: str - BMI状态
            - activity: str - 运动量
            - goal: str - 健康目标
            - allergens: list[str] - 用户过敏原列表
            - user_id: str - 用户ID（用于获取历史记录）

    返回:
        str - 小盘的膳食推荐结果，格式为 "小盘说：..."

    降级返回格式:
        "小盘说：根据您的健康档案，建议您多摄入优质蛋白和蔬菜..."

    示例:
        >>> call_xiaopan_for_meal_plan(user_profile)
        "小盘说：根据您的BMI和减脂目标，建议早餐..."

    调用场景:
        在 /api/chat 接口中，作为 Function Calling 的工具函数被调用
    """
    bmi = user_profile.get('bmi', '--')
    bmi_status = user_profile.get('bmiStatus', '')
    activity = user_profile.get('activity', '--')
    goal = user_profile.get('goal', '--')
    allergens = user_profile.get('allergens', [])
    user_id = user_profile.get('user_id', 'default_user')

    # 获取用户记录（如果可用）
    records = get_user_records(user_id, days=7, limit=5) if user_id else []

    prompt = build_xiaopan_prompt(bmi, bmi_status, activity, goal, allergens, records)

    try:
        response = call_deepseek(prompt, "你是小盘，一位专业的膳食推荐师。")
        return f"小盘说：{response}"
    except Exception as e:
        return f"小盘说：根据您的健康档案，建议您多摄入优质蛋白和蔬菜，控制油脂和碳水摄入。"
