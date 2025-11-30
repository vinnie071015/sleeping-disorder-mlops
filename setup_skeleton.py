import os

# 1. 定义项目根目录
project_root = "/Users/zhengchengsheng/Desktop/sleeping_disorder_prediction"

# 2. 定义要创建的目录结构
folders = [
    "src",              # 源代码目录
    "tests",            # 测试代码目录
    "api",              # API 接口目录
    "docker",           # Docker 配置目录
    "notebooks",        # Notebook 实验目录
    ".github/workflows" # CI/CD 配置目录
]

# 3. 定义要创建的文件 (内容留空，仅做占位)
files = {
    # 4. git 忽略文件 (保留基本配置，防止污染仓库)
    ".gitignore": """
__pycache__/
*.py[cod]
*.so
.ipynb_checkpoints/
.vscode/
.env
venv/
env/
data/
models/
.DS_Store
""",

    # 依赖清单 (空)
    "requirements.txt": "",

    # 5. 源代码包初始化
    "src/__init__.py": "",
    
    # 6. 数据处理文件 (空)
    "src/data_processor.py": "# TODO: 在此编写数据清洗与处理逻辑",

    # 7. 训练脚本文件 (空)
    "src/train.py": "# TODO: 在此编写 SageMaker 训练脚本",

    # 测试包初始化
    "tests/__init__.py": "",

    # 8. 测试脚本 (空 - 后续在此处理 sys.path 和测试逻辑)
    "tests/test_data.py": "# TODO: 在此编写针对 data_processor 的单元测试",

    # 9. API 接口文件 (空)
    "api/app.py": "# TODO: 在此编写 FastAPI 推理接口",

    # 10. Dockerfile (空)
    "docker/Dockerfile": "# TODO: 在此编写 Docker 镜像构建指令"
}

def create_clean_structure():
    # 创建根目录
    if not os.path.exists(project_root):
        os.makedirs(project_root)
        print(f"📁 Created root: {project_root}")
    
    # 创建文件夹
    for folder in folders:
        path = os.path.join(project_root, folder)
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created folder: {folder}")

    # 创建文件
    for filename, content in files.items():
        path = os.path.join(project_root, filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content.strip())
            print(f"📄 Created file: {filename}")
        else:
            print(f"⚠️ Exists: {filename}")

    print(f"\n🎉 文件系统骨架已就绪！位置: {project_root}")

if __name__ == "__main__":
    create_clean_structure()