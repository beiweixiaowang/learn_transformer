# transformer初探

- **Learner** : 永驻
- **Date** : 2021.10.14

2017年谷歌在一篇名为《Attention Is All You Need》的论文中,提出了一个基于attention(自注意力机制)结构来处理序列相关的问题的模型，名为Transformer。**Transformer模型摒弃了固有的定式，并没有用任何CNN或者RNN的结构，而是使用了`Attention注意力机制`，自动捕捉输入序列不同位置处的相对关联，善于处理较长文本，并且该模型可以高度并行地工作，训练速度很快。**

## 模型架构

![模型结构图](img/model_img.jpg)

根据模型架构图，可以看出模型大致分为`**Encoder(编码器)**`和`**Decoder(解码器)**`两个部分，分别对应上图中的左右两部分。

* 其中编码器由N个相同的层堆叠在一起，每一层又有两个子层。第一个子层是一个`Multi-Head Attention`(**多头的自注意机制**)，第二个子层是一个简单的`Feed Forward`(**全连接前馈网络**)。两个子层都添加了一个 **残差连接+layer normalization**的操作。

* 模型的解码器同样是堆叠了N个相同的层，不过和编码器中每层的结构稍有不同。对于解码器的每一层，除了编码器中的两个子层`Multi-Head Attention`和`Feed Forward`，解码器还包含一个子层`Masked Multi-Head Attention`，如图中所示每个子层同样也用了residual +ayer normalization。
* 模型的输入由`Input Embedding`和`Positional Encoding`(**位置编码**)两部分组合而成，模型的输出由Decoder的输出简单的经过softmax得到。
我们下面通过代码，来实现各个模块




## 模型输入
### Embeddings
`Embedding`层的作用是将某种格式的输入数据，例如文本，转变为模型可以处理的向量表示，来描述原始数据所包含的信息。  
`Embedding`层输出的可以理解为**当前时间步的特征**，如果是文本任务，这里就可以是Word Embedding，如果是其他任务，就可以是任何合理方法所提取的特征。
`Embedding`技术在NLP中应用广泛，其中细节不做赘述。
***  
使用torch中Embedding模块来构建:
```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        类的初始化函数
        d_model：指词嵌入的维度
        vocab:指词表的大小
        """
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """
        Embedding层的前向传播逻辑
        参数x：这里代表输入给模型的单词文本通过词表映射后的one-hot向量
        将x传给self.lut并与根号下self.d_model相乘作为结果返回
        """
        embedds = self.lut(x)
        return embedds * math.sqrt(self.d_model)
```
***
### 对Embedding模块进行测试
``` python
embedding = Embeddings(16, 10)
print(embedding)
input_X = torch.randint(0, 10, (1, 5))
print(input_X)
print(input_X.shape)
embedd_X = embedding(input_X)
print(embedd_X)
embedd_X.shape
```
输出如下
``` python
Embeddings(
  (lut): Embedding(10, 16)
)
tensor([[4, 5, 8, 8, 1]])
torch.Size([1, 5])
tensor([[[  9.7116,  -3.2952,   0.5219,  -1.2501,   5.5110,  -2.7993,   3.2012,
            4.9364,   0.3486,  -2.7039,   2.6776,  11.0604,  -6.9241,   3.0686,
           -0.6369,   5.2696],
         [  0.5425,   3.5664,  -2.1441,  -4.5084,   4.4102,   4.9775,   2.0334,
           -0.1824, -10.1114,  -0.7900,  -3.0759,  -7.8962,   0.9425,   0.4546,
           -1.2102,  -2.6575],
         [  1.2518,   2.5849,   2.3994,   9.0222,   5.8251,   2.5319,   1.4701,
            4.1769,  -6.3250,   7.2940,  -0.4277,  -0.3020,  -1.5351,  -6.7384,
           -4.4161,  -4.0305],
         [  1.2518,   2.5849,   2.3994,   9.0222,   5.8251,   2.5319,   1.4701,
            4.1769,  -6.3250,   7.2940,  -0.4277,  -0.3020,  -1.5351,  -6.7384,
           -4.4161,  -4.0305],
         [ -3.5904,  -9.1608,   3.9014,  -1.4941,   6.0639,  -6.8471,   0.7589,
            6.4332,  -2.2029,   3.1529,   4.8142,   7.7268,  11.5662,   6.0759,
           -7.6120,  -7.0156]]], grad_fn=<MulBackward0>)
torch.Size([1, 5, 16])
```
### Positional Encoding
`Positional Encodding`位置编码的作用是为模型提供当前时间步的前后出现顺序的信息。因为Transformer不像RNN那样的循环结构有前后不同时间步输入间天然的先后顺序，所有的时间步是同时输入，并行推理的，因此需要在时间步的特征中融合进位置编码。

**位置编码选择多样，可以是固定的，也可以设置为`可学习的参数`**
我们选择固定的位置编码。使用不同频率的sin和cos函数来进行位置编码，如下所示：

$$PE_{pos,2i}=sin(pos/10000^{2i/d_{model}})$$         
$$PE_{pos,2i+1}=cos(pos/10000^{2i/d_{model}})$$  
可以绘制出这两个函数的图像：

![sin_cos](img/sin_cos_img.jpg)

蓝色为sin, 绿色为cos
***

具体代码实现如下：  


ˋˋˋpython

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        位置编码器类的初始化函数
        
        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

ˋˋˋ


    






