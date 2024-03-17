# 超参数配置

配置文件路径

dtjsp/config.py



需要调的参数

1. 采样数据集

--dataset 

- DRL: 使用nips论文介绍的模型作为behavior policy

- other：尚未实现

  

2. 训练和推理时的上下文长度

```--K```

建议100-200之间，根据效率进行调整

3. 奖赏的缩放规模

```--reward_scale```

暂定100，一般一个环境的总returns-to-go约为-1000。

4. 图网络输出的embedding大小

   ```--hidden_dim```

   原论文64

   

