import torch
from torch import Tensor, nn 
import torch.nn.functional as F 
import numpy as np 


class SelfAttention(nn.Module):
    def __init__(self, input_vector_dim:int, dim_k=None, dim_v=None) -> None:
        """
        初始化SelfAttention，包含以下参数：
        input_vector_dim: 输入向量的维度，对应公式中的d_k。加入我们将单词编码为了10维的向量，则该值为10
        dim_k：矩阵W^k和W^q的维度
        dim_v：输出向量的维度。例如经过Attention后的输出向量，如果你想让它的维度是15，则该值为15；若不填，则取input_vector_dim，即与输入维度一致。
        """
        super().__init__()
        
        self.input_vector_dim = input_vector_dim
        
        # 如果dim_k和dim_v是None，则取输入向量维度
        if dim_k is None:
            dim_k = input_vector_dim
        if dim_v is None:
            dim_v = input_vector_dim
        
        """
        实际编写代码时，常用线性层来表示需要训练的矩阵，方便反向传播和参数更新
        """
        self.W_q = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_k = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_v = nn.Linear(input_vector_dim, dim_v, bias=False)
        
        # 这个是根号下d_k
        self._norm_fact = 1 / np.sqrt(dim_k)
    
    def forward(self, x):
        """ 
        进行前向传播
        x： 输入向量，size为(batch_size, input_num, input_vector_dim)
        """
        # 通过W_q, W_k, W_v计算出Q,K,V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        """
        permute用于变换矩阵的size中对应元素的位置
        即：将K的size由(batch_size, input_num, output_vector_dim) 变为 (batch_size, output_vector_dim, input_num)
        ----
        0,1,2 代表各个元素的下标，即变换前 batch_size所在的位置是0，input_num所在的位置是1
        """
        K_T = K.permute(0, 2, 1)
        
        """ 
        bmm 是batch matrix-matrix product，即对一批矩阵进行矩阵相乘。相比于matmul,bmm不具备广播机制
        """
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self._norm_fact)
        
        """ 
        最后再乘以 V
        """
        output = torch.bmm(atten, V)
        
        return output


def attention(query:Tensor, key:Tensor, value:Tensor):
    """ 
    计算Attention的结果。
    这里其实传入对的是Q，K，V；而Q，K，V的计算是放在模型中的，请参考后续的MultiHeadAttention类。
    
    这里的Q，K，V有两种shape，如果是Self-Attention，shape为(batch, 词数, d_model),
                            例如(1, 7, 128)，表示batch_size为1，一句7个单词，每个单词128维
                            
                            但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
                            例如(1, 8, 7, 16)，表示batch_size为1,8个head，一句7个单词，128/8=16。
                            
                            这样其实也能看出来，所谓的MultiHead其实也就是将128拆开了。
                            
                            在Transformer中，由于使用的是MultiHead Attention，所以Q、K、V的shape只会是第二种。
    """

    """ 
    获取 d_model 的值。之所以这样可以获取，是因为 query 和输入的 shape 相同。
    若为Self-Attention，则最后一维都是词向量的维度，也就是 d_model 的值；
    若为MultiHead-Attention，则最后一维是 d_model/h，h表示head数。
    """
    d_k = query.size(-1)
    
    # 执行QK^T / 根号下d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    """ 
    执行公式中的softmax
    这里的 p_attn 是一个方阵；若为Self-Attention，则shape为(batch, 词数， 词数)；
    若为MultiHead-Attention，则shape为(batch, head数, 词数, 词数)
    """
    p_attn = scores.softmax(dim=-1)
    
    """ 
    最后再乘以 V.
    对于Self-Attention来说，结果 shape 为(batch, 词数, d_model)，这也就是最终的结果了。
    对于MultiHead-Attention来说，结果 shape 为(batch, head数, 词数, d_model/head数)
    而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention该做的事。
    """
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, h:int, d_model:int) -> None:
        """ 
        h: head数
        d_model: d_model数
        """
        super().__init__()
        
        assert d_model % h == 0, "head number should be divided by d_model"
        
        self.d_k = d_model // h
        self.h = h

        # 定义W^q、W^k、W^v和W^o矩阵。
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model)
        ]
    
    def forward(self, x):
        # 获取batch_size
        batch_size = x.size(0)
        
        """ 
        1. 求出Q、K、V。这里是求MultiHead的Q、K、V，所以shape为(batch, head数, 词数, d_model/head数)
            1.1 首先，通过定义的W^q, W^k, W^v 求出Self-Attention的Q、K、V。此时，Q、K、V的shape为(batch, 词数, d_model)
                对应代码为 linear(x)
            1.2 分为多头，即将shape由(batch, 词数, d_model)变为(batch, 词数, head数, d_model/head数)
                对应代码为 .view(batch_size, -1, self.h, self.d_k)
            1.3 最终交换 词数 和 head数 这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数, d_model/head数)
                对应代码为 .transpose(1,2)
        """
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) for linear, x in zip(self.linears[:-1], (x, x, x))]

        """ 
        2. 求出Q、K、V后，通过Attention函数计算出Attention结果。
            这里x的shape为(batch, head数, 词数, d_model/head数)
            self.attn的shape为(batch, head数, 词数, 词数)
        """
        x = attention(query, key, value)
        
        """ 
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数, d_model/head数)再变为(batch, 词数, d_model)
            3.1 首先, 交换 head数 和 词数 维度，结果为 (batch, 词数, head数, d_model/head数)
                对应代码为
        """
        x = x.transpose(1,2).reshape(batch_size, -1, self.h * self.d_k)

        """ 
        4. 最后，通过W^o矩阵再执行一次线性变换，得到最终结果
        """
        return self.linears[-1](x)
        

class PositionalEncoding(nn.Module):
    """
    基于三角函数的位置编码
    """
    def __init__(self, num_hiddens, dropout=0, max_len=1000):
        """
        num_hiddens:向量长度  
        max_len:序列最大长度
        dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建一个足够长的P : (1, 1000, 32)
        self.P = torch.zeros((1, max_len, num_hiddens))
        
        # 本例中X的维度为(1000, 16)
        temp = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(temp)   #::2意为指定步长为2 为[start_index : end_index : step]省略end_index的写法
        self.P[:, :, 1::2] = torch.cos(temp)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)  # torch 加法存在广播机制，因此可以避免batchsize不确定的问题
        return self.dropout(X)
    
 
class FeedForward(nn.Module):
    def __init__(self, d_model:int, hidden_num:int=2048) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, d_model)
        )
    
    def forward(self, x):
        return self.linear(x)        



class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, d_model: int, dropout: float = 0.1):
        """ 
        sublayer: Multi-Head Attention module 或者 Feed Forward module的一个.
        残差连接：上述两个module的输入x和module输出y相加，然后再进行归一化。
        """
        super().__init__()
        
        self.sublayer = sublayer  
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.dropout(self.sublayer(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
     ):
        """ 
        d_model: 词向量维度数
        num_heads: 多头注意力机制的头数
        dim_feedforward: feedforward 模块的隐含层神经元数
        """
        super().__init__()
        
        """ 
        1. 进行多头注意力计算
        """
        self.multi_head_attention_module = Residual(
            sublayer=MultiHeadAttention(h=num_heads, d_model=d_model),
            d_model=d_model,
            dropout=dropout
        )
        
        """ 
        2. 进行前馈神经网络计算
        """
        self.feed_forward_module = Residual(
            sublayer=FeedForward(d_model=d_model, hidden_num=dim_feedforward),
            d_model=d_model,
            dropout=dropout
        )
        
    def forward(self, x:Tensor) -> Tensor:
        # 1. 多头注意力计算
        x = self.multi_head_attention_module(x)
        # 2. 前馈神经网络计算
        x = self.feed_forward_module(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers: int = 6,
        d_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        max_len: int = 1000
    ):
        """ 
        Transformer 编码器
        num_layers: TransformerEncoderLayer 层数
        d_model: 词向量维度数
        num_heads: 多头注意力机制的头数
        dim_feedforward: 前馈神经网络的隐含层神经元数
        dropout: 
        max_len: 三角函数位置编码的最大单词数量，需要设置超过数据集中句子单词长度
        """
        super().__init__()
        """ 
        1. 实例化 num_layers 个TransformerEncoderLayer
        """
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        """ 
        2. 初始化位置编码器
        """
        self.pe = PositionalEncoding(num_hiddens=d_model, max_len=max_len)
        
    def forward(self, x: Tensor) -> Tensor:
        """ 
        x: (batchsize, sequence_number, d_model),sequence_number 表示句子的单词数量，d_model表示每个词的编码维度
        """
        
        """ 
        1. 对输入x添加位置编码信息
        """
        x  = self.pe(x)
        
        """ 
        2. 逐层计算，最后输出特征提取后的values
        """
        for layer in self.layers:
            x = layer(x)
 
        return x

      
if __name__ == '__main__':
    # model = SelfAttention(10)
    # model(torch.rand(3, 16, 10))
    # model = MultiHeadAttention(h=8, d_model=64)
    # model(torch.rand(3, 16, 64))
    dat = torch.rand(3, 16, 64)
    model = TransformerEncoder(num_heads=8, num_layers=6, d_model=64, dim_feedforward=128, dropout=0.1, max_len=1000)
    y = model(dat)
    print(model)

