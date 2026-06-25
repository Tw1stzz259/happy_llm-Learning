#! /bin/bash

stat .venv > /dev/null 2>&1 /dev/null
if [ $? != 0 ]; then
    echo "创建虚拟环境"
    python -m venv .venv
fi

source .venv/bin/activate

echo "检查依赖"
pip install -r requirements.txt

echo "测试依赖"
python -m py_compile code/config.py code/model.py
if [ $? == 0 ]; then
    echo "依赖测试成功"
else
    echo "依赖测试失败，请手动检查"
    exit 1
fi

stat ./code/sft_model_40M > /dev/null 2>&1 /dev/null

if [ $? == 0 ]; then
    stat ./code/sft_model_40M/sft_576_9_6144_final.pth > /dev/null 2>&1 /dev/null
    if [ $? == 0 ]; then
        echo "软件启动中"
        clear
        python code/chat.py
    else
        echo "模型不存在"
        exit 1
    fi
else
    mkdir ./code/sft_model_40M
    echo "模型不存在"
    exit 1
fi
