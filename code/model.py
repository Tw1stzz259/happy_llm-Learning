import torch
import torch.nn as nn
from typing import Tuple

#构建RMSNorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  #权重矩阵初始化全为1

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output*self.weight



def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
        #获取输入张量的形状：批量大小，序列长度，键值对头数量，以及每个头的维度大小
        bs,slen,n_kv_heads,head_dim = x.shape

        if n_rep == 1:
            return x

        return(
            x[:,:,:,None,:] #在第四个维度前添加一个新的维度
            .expand(bs,slen,n_kv_heads,n_rep,head_dim)  #将新添加的维度扩展到n_rep大小，实现重复的效果
            .reshape(bs,slen,n_kv_heads*n_rep,head_dim) #重新塑形
        )

##dim为dim//n_head，对每个head进行旋转嵌入
def precompute_freqs_cis(dim:int,end:int,theta:float = 10000.0):
    freqs = 1.0/(theta ** (torch.arange(0,dim,2)[:(dim//2)].float()/dim))
    t = torch.arange(end,device=freqs.device)
    freqs = torch.outer(t,freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos,freqs_sin


def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    #获取x的维度数
    ndim = x.ndim
    #断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    #断言，确保freqs_cis的形状与x的第二维度和最后一维相同
    assert freqs_cis.shape == (x.shape[1],x.shape[-1])

    #构造一个新的形状，除了第二维和最后一维，其他的维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    #将freqs_cis调整为新的形状,并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
