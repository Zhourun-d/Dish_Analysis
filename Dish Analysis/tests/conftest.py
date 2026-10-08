"""
pytest 全局配置与公共夹具。

三个关键设计:

1. 顶替 ultralytics
   analyze.py 在**导入阶段**就会执行 YOLO("./model/best.pt")，真实加载需要
   torch（数百 MB 到数 GB）与两个 .pt 权重文件。而本套测试覆盖的是接口行为与
   业务逻辑，不覆盖模型推理本身，因此默认用一个最小桩件顶替 ultralytics，
   让测试在任意机器上秒级跑完。
   若想在真实模型下运行: 设环境变量 USE_REAL_MODELS=1（需已装 torch 且权重文件就位）。

2. 隔离数据库
   每个用例把 services.DB_PATH 指向 tmp_path 下的临时文件，绝不触碰项目里的 records.db。

3. 不依赖 .env
   夹具会注入一个假密钥，使测试不因缺少 DEEPSEEK_API_KEY 而失败。
"""

import os
import sys
import types

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

# 让 `import analyze` / `import services` 可用
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# analyze.py 读取模型与 Excel 用的都是 ./ 相对路径，切到项目根目录
os.chdir(PROJECT_ROOT)


def _install_ultralytics_stub():
    if os.environ.get("USE_REAL_MODELS") == "1":
        return
    if "ultralytics" in sys.modules:
        return

    stub = types.ModuleType("ultralytics")

    class YOLO:
        def __init__(self, path, *args, **kwargs):
            self.path = path
            self.names = {}

        def __call__(self, source, *args, **kwargs):
            raise NotImplementedError(
                "测试环境未加载真实模型，请先替换 analyze.model / analyze.multi_model"
            )

    stub.YOLO = YOLO
    sys.modules["ultralytics"] = stub


_install_ultralytics_stub()


@pytest.fixture(scope="session")
def analyze_module():
    import analyze
    return analyze


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """把数据库指向临时文件，保证用例之间互不影响。"""
    import services

    monkeypatch.setattr(services, "DB_PATH", str(tmp_path / "records_test.db"))
    services.init_db()
    return services.DB_PATH


@pytest.fixture()
def client(analyze_module, temp_db, monkeypatch):
    """Flask 测试客户端（已注入假密钥、已隔离数据库）。"""
    import services

    monkeypatch.setattr(analyze_module, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(services, "DEEPSEEK_API_KEY", "test-key")
    return analyze_module.app.test_client()
