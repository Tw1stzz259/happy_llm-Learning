# happy-llm Learning

这是一个用于学习和实现 Happy-LLM 教程内容的个人学习项目。当前代码主要围绕第五章“动手搭建大模型”，逐步实现模型配置、RMSNorm、RoPE、Attention、MLP 和 DecoderLayer 等基础模块。

## 快速启动

使用脚本快速安装依赖并启动对话程序

```bash
./start.sh
```

## 模型对话演示

仓库不包含模型权重文件。对话脚本默认读取已经完成 SFT 的 40M 参数模型权重：

```text
code/sft_model_40M/sft_576_9_6144_final.pth
```

下载模型权重后，请把权重文件放到上面的路径。如果目录不存在，先创建目录：

```bash
mkdir ./code/sft_model_40M
```

运行连续对话：

```bash
python ./code/chat.py
```

启动后在 `User:` 后输入问题并回车。退出时输入空行、`/exit` 或 `/quit`。

也可以一次性提问：

```bash
python. code/chat.py --prompt "你好，请介绍一下你自己。"
```

如果在其他电脑演示，至少需要保留以下文件：

```text
code/chat.py
code/config.py
code/model.py
code/Tokenizer/
code/sft_model_40M/sft_576_9_6144_final.pth
```

如果权重文件放在其他位置，可以用 `--ckpt` 指定：

```bash
python ./code/chat.py --ckpt "/path/to/sft_576_9_6144_final.pth"
```

注意：这是从零训练的小参数量模型，适合演示训练链路和基础对话形式，不适合用作可靠的事实问答或数学推理模型。

## 当前环境

本项目推荐使用本地 linux 虚拟环境 `.venv`。

```bash
source .venv/bin/activate
```

也可以不激活环境，直接使用虚拟环境中的 Python：

```bash
.venv/scripts/python
```

当前已安装的核心依赖：

- `torch 2.12.0+cpu`
- `transformers 5.10.2`

## 运行检查

```bash
.venv/scripts/python -m py_compile code/config.py code/model.py
```
