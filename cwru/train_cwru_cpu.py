"""
基于 CPU 的 CWRU 轴承故障诊断训练脚本。

- 仅使用 CPU 进行训练，便于在没有 NVIDIA GPU 的环境中运行。
- 示例使用 4 个故障类别：
  - 正常：97.mat
  - 内圈故障：105.mat
  - 滚动体故障：118.mat
  - 外圈故障：130.mat

本脚本会将长序列振动信号切分成固定长度的窗口，以便训练一维卷积网络。
"""
from __future__ import annotations  #annotation注解
import argparse              #argparse参数解析
import os                     #os模块提供了一种与操作系统交互的方式
from typing import Dict, List, Tuple  #从typing模块导入Dict（字典）、List（列表）、Tuple（元组）
import numpy as np            #numpy用于数组操作和数值计算，是Python中用于科学计算的基础库
import torch                  #导入PyTorch，是Python中用于深度学习的库，提供张量计算和自动微分功能
from scipy.io import loadmat  
#从scipy.io导入loadmat，用于读取MATLAB的.mat文件，scipy.io是SciPy的输入/输出模块，用于读写各种科学数据格式。
#scipy（Scientific Python）是基于numpy(Numerical Python)的高级科学计算算法库，
#numpy是基础的核心的（索引、切片、形状变换、加减乘除、三角函数等），scipy是高级的（优化、统计、信号处理等）。
from torch import nn         
#为什么上面import torch了，这里还要from torch import nn？
#from torch import nn 其实就是 import torch.nn as nn，这样做一是后面写起来简洁方便
#其二，看到nn.XXX就知道是“网络层/损失函数...”等在深度学习中与神经网络相关的模块
#torch除了torch.nn模块，还有torch.optim（训练相关）、torch.utils.data（数据相关）、torch.cuda / torch.device（设备）等模块
from torch.utils.data import DataLoader, Dataset 
#torch.utils.data用于数据加载和处理  torch.utils.data是torch中的一个模块，用于数据加载和处理
#DataLoader：数据加载器，用于批量加载和迭代数据；Dataset：数据集基类，用于自定义数据集
device = torch.device("cpu") #torch.device("cpu")表示使用cpu设备  torch.device是torch中的一个类，用于创建设备

class CWRUDataset(Dataset):   # CWRUDataset类继承自Dataset类（line30）；括号里是父类
    """简单的 CWRU 数据集封装；CWRUDstaset这个类用于封装CWRU数据集的处理逻辑
    封装：将相关的数据（属性）和操作（方法）组合在一起，形成一个独立的单元（类），并隐藏内部实现细节，只对外提供必要的接口。
    电视机：内部电路复杂，但只需按遥控器按钮即可使用；汽车：内部有引擎、变速箱等，但只需踩油门、转方向盘即可驾驶
    - 读取 .mat 文件中的振动信号（默认使用带 `DE_time` 的键）。
    - 将长序列按滑动窗口切分成若干样本，节省内存并增加样本数量。
    """
    """
    下述def init是在：
    增加样本数量：一个长信号可生成多个训练样本;
    节省内存：按需加载，避免一次性加载所有数据
    适配模型：将长序列切分为固定长度，便于 CNN 处理; 
    这是数据预处理的核心步骤，将原始振动信号转换为模型可用的训练样本。
    """
    def __init__(          #def用来定义函数或方法    def 函数名(参数)：
                           #                         函数体
                           #                         return 返回值
                           # _init_是特殊方法名（构造函数），用于初始化对象 
        self,
        root_dir: str, #数据文件所在目录
        file_labels: Dict[str, int], #文件名到标签的映射，如{"97.mat": 0,"105.mat": 1}
        window_size: int = 2048, #滑动窗口大小
        step_size: int = 1024, #滑动窗口步长
    ) -> None: 
        self.samples: List[Tuple[np.ndarray, int]] = [] #创建空列表，用于存储（信号窗口，标签）元组
        for file_name, label in file_labels.items():
            file_path = os.path.join(root_dir, file_name) #构建文件路径，os.path.join方法将目录和文件名拼接成一个路径
            signal = self._load_signal(file_path) #self._load_signal方法加载信号
            windows = self._window_signal(signal, window_size, step_size) #将长信号按滑动窗口切分成多个短窗口
            self.samples.extend((win, label) for win in windows) #将每个短窗口与对应标签组成元组，添加到samples列表中

    @staticmethod     # @staticmethod是装饰器，表示_load_signal方法是一个静态方法
                      # 静态方法不需要实例化类就可以调用，，，，静态方法第一个参数不需要self
    def _load_signal(file_path: str) -> np.ndarray: #_load_signal方法加载信号，单下划线开头代表内部方法
                                                    #内部方法（私有方法）是类内部使用的辅助方法，通常不供外部直接调用。
        """加载单个 .mat 文件并返回一维信号。
        数据预处理的第一步，将 MATLAB 格式的振动信号转换为 Python 可用的数组格式。"""
        if not os.path.exists(file_path): #如果文件不存在，则抛出FileNotFoundError异常
            raise FileNotFoundError(f"未找到数据文件: {file_path}")
        mat_data = loadmat(file_path) #loadmat方法加载mat文件
        data_key = next((k for k in mat_data.keys() if "DE_time" in k), None) #next方法找到DE_time键
                                                                              #next():返回生成器第一个匹配项，未找到返回None
        signal = mat_data[data_key].squeeze() #squeeze方法将信号压缩（确保是一维信号）
        return signal.astype(np.float32) #astype方法将信号转换为float32类型，统一数据类型，符合pytorch常用形式

    @staticmethod
    def _window_signal(signal: np.ndarray, window_size: int, step_size: int) -> List[np.ndarray]: 
        #_window_signal方法将信号切分成窗口
        """将长信号按窗口切分。"""
        windows: List[np.ndarray] = [] #windows是窗口列表
        for start in range(0, len(signal) - window_size + 1, step_size): 
                   # range(start, stop, step)：从 start 开始，每次加 step，到达但不包含 stop
            end = start + window_size  # 计算当前窗口的结束索引
            windows.append(signal[start:end])  # append方法将窗口添加到窗口列表中
        return windows #返回窗口列表

    def __len__(self) -> int:  # noqa: D401 - 简单长度返回 #__len__方法返回数据集长度
        return len(self.samples)  #返回内部样本列表的长度，也就是数据集中有多少个窗口样本。

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window, label = self.samples[idx] #self.samples[idx]是数据集的样本  window是窗口信号(numpy数组)，label是标签
        tensor = torch.from_numpy(window).unsqueeze(0) # torch.from_numpy方法将窗口转换为torch张量 
        # .unsqueeze(0)在第0维添加一个维度，把形状从 [L] 变为 [1, L]，以符合一维卷积输入的通道格式
        return tensor, label #返回tensor和label
                             #__len____getitem____init__都是隐式调用，都是特殊方法


class SimpleCNN(nn.Module): #SimpleCNN类继承自nn.Module类 
                            #nn.Module类是torch.nn模块中的一个类，用于创建神经网络 模型
    """轻量级一维卷积分类器，适合 CPU 训练。"""
    def __init__(self, num_classes: int) -> None: #__init__方法初始化SimpleCNN类
        super().__init__()  # super()代表“父类”（也就是nn.Module）;super().__init__()调用父类的构造函数
        self.features = nn.Sequential( 
                     #给当前对象挂一个属性 self.features,这是一个子网络（特征提取部分）
                     #nn.Sequential(): python提供的顺序容器，容器中包含多个神经网络层
                     # y=self.features(x)，相当于x先过第一层→结果再过第二层→…→过完最后一层。  
        ### 下面就是features这个子网络里的一层层：
            nn.Conv1d(1, 16, kernel_size=16, stride=2, padding=7), 
            # 一维卷积层Conv1d，专门处理一维信号，参数含义：in_channels=1:输入只有一个通道（原始振动信号）
            # out_channels=16：输出16个通道，可以理解为学到16组不同的滤波器
            # kernel_size=16：卷积核长度16个点，相当于一个16点滑动窗口
            # stride=2：步长为2，每次滑动2个采样点→时间长度减半
            # padding=7：两端各补7个点，控制输出长度（大致保持“卷积+步长”后仍比较平滑） 
            # 类似16个FIR滤波器，对振动信号做局部特征提取和下采样。
            nn.BatchNorm1d(16), #一维batch normalization，通道数16（对应上一层输出的16通道）
                                #作用：对每个通道的特征做标准化（减均值除方差），再学两个可训练参数做线性变换
                                #好处：训练更稳定，收敛更快，缓解梯度消失/爆炸
            nn.ReLU(), #非线性激活函数：ReLU(x) = max(0,x)
                       #给网络带来非线性能力，能拟合更复杂的函数；同时计算简单、梯度不错，是卷积网络里最常见的激活。
            nn.MaxPool1d(kernel_size=2), #一维最大池化，窗口大小为2（stride 默认等于kernel_size=2）
                                         #再次把时间长度减半，保留局部窗口中的最大值；
                                         # 起到下采样 + 提取显著特征的效果；
                                         # 也一定程度上带来平移不变性（局部稍微移动，最大值不变）。
        ### 以下4行为第二个卷积块
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=3), #输入通道从16提升到32，学到更高级、更抽象的特征
                                                                   # 卷积核更短，但仍在做下采样
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2),
        ### 第二个卷积块继续在时间维度上缩短长度，同时提升通道数（特征维度），"原始波形"逐渐变成"故障模式特征"。
        ### 以下为第三个卷积块,通道数32->64,卷积核更短(4点),更关注局部模式(如高频冲击),stride=2继续下采样
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), # 这个时候，时间长度已经被多次“减半”，通道数已增加到64个特征通道
            nn.AdaptiveAvgPool1d(1), #自适应平均池化：不管输入的时间长度是多少，输出的长度都变成1
                                     #对每个通道在时间维上求平均，得到一个标量 → 相当于做“全局平均池化”
                                     #输出形状变成[batch_size, 64, 1]，64是通道数，1是“压缩后的时间维”
                                     #不再依赖固定输入长度（即使以后窗口长度改一点，也能适配）
                                     #自然地把“序列特征”汇总成一个固定长度的向量，为后面的全连接分类做准备
        )
        self.classifier = nn.Sequential(              
            # self.classifier:分类器部分（全连接层头部），负责把64维特征映射到最终的类别logits
            nn.Flatten(), # 展平层; Flatten:展平操作，把除了 batch 维以外的所有维度压成一维
                          # 输入是 [batch_size, 64, 1]；展平后变成 [batch_size, 64]（因为 64×1 = 64）
                          # 为啥要展平？因为 nn.Linear 期望输入是 [batch_size, 特征维度] 的 2D 形状
            nn.Linear(64, 64), #全连接层：输入 64 维，输出 64 维
                               #可以理解为一个“特征混合层”，让 64 维特征之间做一次线性组合，提升表达能力
            nn.ReLU(), #给全连接层后的特征加非线性；防止整个网络退化成大号线性变换
            nn.Dropout(0.3), #Dropout 层，丢弃概率 p=0.3：训练时，每次随机把 30% 的神经元输出置零；
                                                        #测试/验证时会自动关闭（这就是前面 .train()/.eval() 的差别）。
                            #防止过拟合，让模型不要过度依赖某几个特征；增加“随机扰动”，相当于多模型集成的一种近似。
            nn.Linear(64, num_classes), #最后一层线性层：输入 64 维，输出 num_classes 维：
                                        #这里 num_classes = 4（正常 + 内圈 + 滚动体 + 外圈）。
                            #输出形状是[batch_size,4]，每一行对应一个样本的4个类别的“得分”(logits)：
                            #训练时配合nn.CrossEntropyLoss使用；推理时可以softmax取概率，或者argmax选最大得分的类别
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义网络的前向传播逻辑
                                                         # x是输入张量，类型注解说明它是 torch.Tensor
                                                         # 返回值注解同样是 torch.Tensor（输出 logits）
        return self.classifier(self.features(x)) #返回分类器输出
                                                 #先把 x 丢进 self.features 做特征提取：
                                                 # h = self.features(x)，得到形状大致 [batch_size, 64, 1]
                                                 # 再把提取后的特征丢进 self.classifier 做分类：
                                                 # y = self.classifier(h)，得到 [batch_size, num_classes]。
                                                 # 一行写在一起就是：“先 features，再 classifier”。
                                                 # 这是非常典型的“骨干网络 + 分类头”结构。

def train_one_epoch(        # 定义函数，名叫train_one_epoch
    model: nn.Module,       # model是一个神经网络模型对象，必须继承自nn.Module
    dataloader: DataLoader, # dataloader是一个数据加载器对象，来自torch.utils.data.DataLoader
                            # 它负责一“批”（batch）一“批”地从数据集中取出(inputs, labels)给你        
    criterion: nn.Module,   # 损失函数对象（也是一个nn.Module）；你的main里面写的是：criterion=nn.CrossEntropyLoss()，所以这里实际传进来的就是交叉熵损失。
    optimizer: torch.optim.Optimizer,   # 优化器对象，比如torch.optim.Adam(...)；它负责根据梯度更新模型参数，是“学习”的执行者。
) -> float:                             # 返回值的类型注解：这个函数会返回一个float浮点数，具体就是：这一轮epoch的平均损失（average loss）
    """单轮训练循环，返回平均损失。"""
    model.train()    # 把模型切换到 训练模式（training mode）
                     # 切换到训练模式是因为很多层在训练/测试时行为不一
                     # 如nn.Dropout训练时随机把一部分神经元输出置零（防止过拟合）；测试时要关闭Dropout，不能再乱丢
                     # 如nn.BatchNorm1d训练时用当前batch的均值/方差来更新内部统计；测试时用“历史移动平均”的均值/方差
                     # 如果不 model.train()，模型可能处于 eval 状态，训练出来的参数就不对劲
    total_loss = 0.0 # 初始化“总损失”，建一个浮点变量 total_loss，用来累加所有batch的损失和
                     # 思路为：这一轮epoch要遍历所有样本，每个batch都会算一个loss，把这些loss按样本数加和，最后再除以总样本数→就是平均损失
    for inputs, labels in dataloader:  # 使用 for 循环从 dataloader 里一批一批地拿数据
        # inputs：一个 batch 的输入张量，形状大概是 [batch_size, 1, window_size]（比如 [64, 1, 2048]）
        # labels：一个 batch 的标签张量，形状是 [batch_size]（长度 64，每个是 0~3）。
        inputs, labels = inputs.to(device), labels.to(device)  # 数据搬到指定设备
        optimizer.zero_grad()  # 清空旧梯度，所有模型参数的grad清零（PyTorch里的梯度是累加的）
        outputs = model(inputs) # 前向传播，pytorch内部机制触发的，model(inputs)等价于model.forward(inputs)
                                # 调用SimpleCNN.forward(self, inputs)
                                # outputs 的形状一般是 [batch_size, num_classes]，比如 [64, 4]；
                                # 每一行是对应样本对每个类别的得分（logits）但我查到的logits是"逻辑值"的意思。
        loss = criterion(outputs, labels) # criterion()在main()中有定义，outputs在上面，labels为真实标签，形状为[batch_size]如[64]），每个元素是0~3的整数类别
        loss.backward()     # 反向传播   
                            # 在前向传播时，PyTorch 自动构建计算图：
                            # inputs → model → outputs → loss
                            #            ↓
                            #        所有中间层（Conv1d, BatchNorm, Linear等）

                            # loss.backward() 使用链式法则从损失开始，逐层反向计算梯度： 
                            # ∂loss/∂loss = 1  (起点)
                            #      ↓
                            # ∂loss/∂outputs  (损失对输出的梯度)
                            #      ↓
                            # ∂loss/∂classifier参数  (损失对分类器参数的梯度)
                            #      ↓
                            # ∂loss/∂features参数  (损失对特征提取层参数的梯度)
                            #      ↓
                            # ...逐层传播...                                                                           
        optimizer.step()    # 根据计算出的梯度，按优化算法更新模型参数
                            # optimizer在main()中有定义。.step是优化器执行一次参数更新的动作名
        total_loss += loss.item() * inputs.size(0) 
                            # loss 默认是当前batch的“平均损失”；loss.item() 取出其标量值
                            # inputs.size(0)是batch内样本数(batch_size)，两者相乘得到“该batch的损失总和”（平均损失×样本数）
    return total_loss / len(dataloader.dataset) 
                            # 计算整个epoch的平均损失，从而做到按样本数加权的准确平均

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """在验证集上评估损失与准确率。"""
    model.eval()       # 切换为评估模式
    total_loss = 0.0    
    correct = 0        # 预测正确样本数清零
    with torch.no_grad():  # 关闭梯度计算
        for inputs, labels in dataloader:   # 按批次遍历验证集
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到指定设备（CPU）
            outputs = model(inputs)                                                                    
            loss = criterion(outputs, labels) 
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)   # 取每个样本得分最高的类别索引作为预测
            correct += (preds == labels).sum().item()   # 统计本批预测正确的样本数并累加
    avg_loss = total_loss / len(dataloader.dataset)     # 用总损失除以验证集总样本数，得到整个验证集的平均损失
    accuracy = correct / len(dataloader.dataset)        # 正确样本数除以总样本数，得到准确率
    return avg_loss, accuracy 


def parse_args() -> argparse.Namespace: #parse_args方法解析命令行参数
    parser = argparse.ArgumentParser(description="CPU 版 CWRU 故障诊断训练脚本") 
    parser.add_argument("--data_dir", type=str, default=r"D:\cursorcode\pytorchCWRU311\data\cwru", help="包含 .mat 文件的数据目录") 
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小") 
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数") 
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率") 
    parser.add_argument("--window_size", type=int, default=2048, help="滑动窗口长度") 
    parser.add_argument("--step_size", type=int, default=1024, help="滑动窗口步长") 
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例，剩余用于验证",
    ) #parser.add_argument方法添加一个参数
    return parser.parse_args() #parser.parse_args()方法解析命令行参数


def main() -> None: #main方法主函数  None是空类型
    args = parse_args() #parse_args方法解析命令行参数

    # 固定随机种子，便于复现
    torch.manual_seed(42) #torch.manual_seed方法设置随机种子
    np.random.seed(42) #np.random.seed方法设置随机种子

    # 数据文件与类别映射，用户可自行修改文件名
    file_labels = { #file_labels是数据文件与类别映射
        "97.mat": 0,  # 正常
        "105.mat": 1,  # 内圈故障
        "118.mat": 2,  # 滚动体故障
        "130.mat": 3,  # 外圈故障
    } #file_labels是数据文件与类别映射

    # 加载完整数据集
    dataset = CWRUDataset( #CWRUDataset类继承自Dataset类   Dataset类是torch.utils.data模块中的一个类，用于创建数据集
        root_dir=args.data_dir,
        file_labels=file_labels,
        window_size=args.window_size,
        step_size=args.step_size,
    ) 

    # 划分训练与验证集
    total_size = len(dataset) #total_size是数据集大小                                                                                   
    train_size = int(total_size * args.train_ratio) #train_size是训练集大小
    val_size = total_size - train_size #val_size是验证集大小
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42) 
        # torch.Generator().manual_seed(42)方法设置随机种子
    )   # torch.utils.data.random_split方法将数据集随机分成训练集和验证集

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True) 
    # DataLoader类是torch.utils.data模块中的一个类，用于创建数据加载器
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=len(file_labels)).to(device) #SimpleCNN类继承自nn.Module类   nn.Module类是torch.nn模块中的一个类，用于创建神经网络模型
    criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss方法创建一个交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #torch.optim.Adam方法创建一个Adam优化器

    # 训练主循环
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer) #train_one_epoch方法训练模型
        val_loss, val_acc = evaluate(model, val_loader, criterion) #evaluate方法在验证集上评估损失与准确率
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" 
            #f"Epoch {epoch:02d}/{args.epochs} | "是格式化字符串  train_loss:.4f是训练损失  val_loss:.4f是验证损失  val_acc:.4f是验证准确率                                                                    
        )

    # 最终保存模型参数
    os.makedirs("./checkpoints", exist_ok=True) #os.makedirs方法创建一个目录
    save_path = os.path.join("./checkpoints", "cwru_cnn_cpu.pth") #os.path.join方法将目录和文件名拼接成一个路径
    torch.save(model.state_dict(), save_path) #torch.save方法保存模型参数                                                               
    print(f"模型已保存到: {save_path}") #f"模型已保存到: {save_path}"是格式化字符串  save_path是模型保存路径

if __name__ == "__main__": #__name__是模块名    __main__是主模块名            main方法主函数  None是空类型
    main() #main方法主函数  None是空类型          main方法主函数  None是空类型   