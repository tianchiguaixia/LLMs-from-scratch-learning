# 中文注释说明：
# ​GPTDatasetV1: 该类用于将文本数据转换为适合训练的数据集。它使用滑动窗口将文本分割为重叠的序列，并生成输入和目标序列。
# ​create_dataloader_v1: 该函数用于创建数据加载器，方便在训练时批量加载数据。
# ​MultiHeadAttention: 该类实现了多头注意力机制，允许模型在处理序列时关注不同的部分。
# ​LayerNorm: 该类实现了层归一化，有助于稳定训练过程。
# ​GELU: 该类实现了GELU激活函数，常用于Transformer模型中。
# ​FeedForward: 该类实现了前馈神经网络，通常用于Transformer块中的非线性变换。
# ​TransformerBlock: 该类实现了Transformer块，包含多头注意力和前馈网络。
# ​GPTModel: 该类实现了GPT模型，包含词嵌入、位置嵌入、多个Transformer块和输出层。
# ​generate_text_simple: 该函数用于生成文本，通过逐步预测下一个token来生成连续的文本。
# ​main: 该函数用于测试模型，生成一段文本并输出结果。




import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# 第2章
#####################################

# GPTDatasetV1 类用于处理文本数据并将其转换为适合训练的数据集
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列的token ID
        self.target_ids = []  # 存储目标序列的token ID

        # 使用tokenizer对文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割为重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)  # 返回数据集的长度

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 返回指定索引的输入和目标序列


# create_dataloader_v1 函数用于创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第3章
#####################################

# MultiHeadAttention 类实现了多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义线性层用于生成查询、键和值
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 用于合并多头输出的线性层
        self.dropout = nn.Dropout(dropout)
        # 注册一个因果掩码，用于防止模型看到未来的信息
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 生成键、查询和值
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 将矩阵分割为多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置矩阵以方便计算
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 每个头的点积

        # 将因果掩码转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多头输出
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影

        return context_vec


#####################################
# 第4章
#####################################

# LayerNorm 类实现了层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 平移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# GELU 类实现了GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# FeedForward 类实现了前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# TransformerBlock 类实现了Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 注意力块的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        # 前馈网络的快捷连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 将原始输入加回来

        return x


# GPTModel 类实现了GPT模型
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层后的dropout

        # 多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# generate_text_simple 函数用于生成文本
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文的索引数组，形状为 (B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过支持的最大长度，则裁剪
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取具有最高logits值的词汇索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引追加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


# main 函数用于测试模型
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头的数量
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # dropout率
        "qkv_bias": False        # 查询-键-值是否使用偏置
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # 禁用dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()