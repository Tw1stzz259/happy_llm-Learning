# happy-llm Learning

这是一个用于学习和实现 Happy-LLM 教程内容的个人学习项目。当前代码主要围绕第五章“动手搭建大模型”，逐步实现模型配置、RMSNorm、RoPE、Attention、MLP 和 DecoderLayer 等基础模块。

## 当前环境

本项目推荐使用本地 Windows 虚拟环境 `.venv-win`。

```powershell
.\.venv-win\Scripts\Activate.ps1
```

也可以不激活环境，直接使用虚拟环境中的 Python：

```powershell
.\.venv-win\Scripts\python.exe
```

当前已安装的核心依赖：

- `torch 2.12.0+cpu`
- `transformers 5.10.2`

## 运行检查

```powershell
.\.venv-win\Scripts\python.exe -m py_compile code\config.py code\model.py
```

注意：虚拟环境目录 `.venv/` 和 `.venv-win/` 不应提交到 Git。
