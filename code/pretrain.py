import os
import argparse
import time
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from config import ModelConfig
from model import Transformer
from dataset import PretrainDataset

def Logger(context):
    print(context)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    

def get_lr(it,all):
    """
    计算当前迭代的学习率，使用余弦退火调度策略
    学习率调度策略：
    1.Warmup阶段：学习率从0线性增上到目标学习率
    2.余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3.退出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
    warmup_iters = args.warmup_iters #预热迭代次数
    lr_decay_iters = all #学习率衰减的总迭代次数
    min_lr = args.learning_rate /10  #最小学习率，为初始学习率的1/10
    
    #Warmup阶段：线性增长
    if warmup_iters > 0 and it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    #超出训练步数：保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    #余弦退火阶段
    decay_iters = max(lr_decay_iters - warmup_iters, 1)
    decay_ratio = (it - warmup_iters) / decay_iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #余弦系数
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch):
    """
    训练一个epoch的函数
    
    实现了完整的训练循环，包括：
    1. 数据加载和设备转移
    2. 动态学习率调整
    3. 前向传播和损失计算
    4. 梯度累积和反向传播
    5. 梯度裁剪和优化器更新
    6. 日志记录和模型保存
    
    Args:
        epoch (int): 当前epoch编号
    """
    start_time = time.time()  #记录开始时间
    
    #遍历数据加载器中的每个batch
    for step,(X,Y,loss_mask) in enumerate(train_loader):
        #将数据转移到指定设备
        X = X.to(args.device)  #输入序列
        Y = Y.to(args.device)  #目标序列
        loss_mask = loss_mask.to(args.device)   #损失掩码，用于忽略padding token
        
        #计算当前步骤的学习率
        lr = get_lr(epoch * iter_per_epoch + step,args.epochs * iter_per_epoch)
        #更新优化器中所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        #使用混合精度上下文
        with ctx:
            #前向传播
            out = model(X,Y)
            #计算损失并除以累计步数（用于梯度累计）
            loss = out.last_loss / args.accumulation_steps
            #将loss_mask展平为一维
            loss_mask = loss_mask.view(-1)
            #应用掩码计算有效损失（忽略padding位置）
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            
        #使用scaler进行混合精度的反向传播
        scaler.scale(loss).backward()
        
        #每accumulation_steps步执行一次优化器更新
        if (step+1) % args.accumulation_steps == 0:
            #取消梯度缩放，准备梯度裁剪
            scaler.unscale_(optimizer)
            #梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            
            #执行优化器步骤
            scaler.step(optimizer)
            #更新scaler的缩放因子
            scaler.update()
            
            #清零梯度，set_to_none=True可以节省内存
            optimizer.zero_grad(set_to_none=True)
            
        #每log_interval步记录一次日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # 打印训练进度信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复真实的loss值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
        # 每save_interval步保存一次模型
        if (step + 1) % args.save_interval == 0:
            model.eval()  # 切换到评估模式
            # 构建检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'
            torch.save(model.state_dict(), ckp)
            model.train()  # 切换回训练模式
            
        # 每20000步保存一个带步数标记的检查点
        if (step + 1) % 20000 == 0:
            model.eval()
            # 构建带步数的检查点文件名
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # 保存模型状态字典
            torch.save(model.state_dict(), ckp)
            model.train()


def init_model():
    """
    初始化模型和分词器
    
    功能包括：
    1. 加载预训练的分词器
    2. 创建Transformer模型
    3. 设置多GPU并行训练（如果可用）
    4. 将模型移动到指定设备
    5. 统计并打印模型参数量
    
    Returns:
        tuple: (model, tokenizer) 初始化后的模型和分词器
    """
    def count_parameters(model):
        """
        统计模型中可训练参数的数量
        
        Args:
            model: PyTorch模型
            
        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 从本地路径加载预训练的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is not None:
        lm_config.pad_token_id = tokenizer.pad_token_id

    # 根据配置创建Transformer模型
    model = Transformer(lm_config)

    # 将模型移动到指定设备（GPU或CPU）
    model = model.to(args.device)
    
    # 计算并打印模型参数量（以百万为单位）
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer
    
    

if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--dim", type=int, default=1024, help="模型维度")
    parser.add_argument("--n_layers", type=int, default=18, help="Transformer层数")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--n_kv_heads", type=int, default=8, help="KV注意力头数")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    
    # 实验跟踪和数据加载参数
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="训练数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./Tokenizer", help="tokenizer path")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    

    args = parser.parse_args()
    
    #==================== 模型配置 ====================
    #定义语言模型的配置参数
    lm_config = ModelConfig(
        dim=args.dim,       #模型维度
        n_layers=args.n_layers,    #Transformer层数
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.max_seq_len,
    )
    
    # ==================== 训练环境设置 ====================
    max_seq_len = lm_config.max_seq_len   #最大序列长度
    
    args.save_dir = resolve_path(args.out_dir)  # 模型保存目录
    args.data_path = resolve_path(args.data_path)
    args.tokenizer_path = resolve_path(args.tokenizer_path)
    os.makedirs(args.save_dir, exist_ok=True)
    
    #设置随机种子确保结果可复现
    torch.manual_seed(42)
    
    #确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    
    #设置混合精度训练的上下文管理器
    #cpu训练时使用nullcontext,gpu训练时使用autocast
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast("cuda", dtype=ptdtype)
    
    # ==================== 模型和数据初始化 ====================
    # 初始化模型和分词器
    model,tokenizer = init_model()
    
    #创建训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    #创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  #批次大小
        pin_memory=(device_type == "cuda"),             #将数据加载到固定内存中,加速GPU传输
        drop_last=False,             #不丢弃最后一个不完整的批次
        shuffle=True,                #随机打乱数据
        num_workers=args.num_workers #数据加载的并行工作进程数
    )
    
    # ==================== 优化器和训练组件初始化 ====================
    # 初始化混合精度训练的梯度缩放器
    # 只有在使用float16或bfloat16时才启用
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and args.dtype == "float16"))
    
    # 初始化Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== 开始训练 ====================
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch)

    final_ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_final.pth'
    torch.save(model.state_dict(), final_ckp)
    Logger(f"Final model saved to {final_ckp}")
