# 九章算法 - 强化学习 - 实战项目2  

## 安装过程   
1. 确认已经安装conda, 安装详情请参考之前的教程：https://www.jiuzhang.com/tutorial/ai-camp/477
2. 打开一个terminal，windows用户建议打开Anaconda Prompt.
3. 输入命令行: conda create --name rl_p2 python=3.5
3. 输入命令行: conda activate rl_p2        
如果激活环境遇到了问题，conda会自己提示解决方案，比如：   
、、、
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
If your shell is Bash or a Bourne variant, enable conda for the current user with

    $ echo ". /Users/Andrew/anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
、、、         
NOTE： 注意Andrew是我的用户名，请根据自己的提示信息进行修改 
6. 输入命令行: git clone https://github.com/jiuzhangjiangzuo/reinforcement_learning_p2.git 或直接下载压缩包[link](https://github.com/jiuzhangjiangzuo/reinforcement_learning_p2/archive/master.zip)
7. 输入命令行: cd reinforcement_learning_p2
8. 输入命令行: pip install -r requirements.txt
9. 输入命令行: pip install gym[atari]
10. 安装ffmpeg库       
    Mac OS - 输入命令行: brew install ffmpeg         
             如果Mac OS系统上没有安装brew，请先运行以下命令(参考[brew官网](https://brew.sh/)):    
             `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`            
    Linux - 输入命令行: sudo apt-get install ffmpeg      
    Windows: 请尝试按照官网的说明安装: https://www.ffmpeg.org/download.html       
9. 测试是否可以运行: python dqn_atari.py        
Note: 如果看到一下这个错误，说明环境是没有问题的，这个错误是因为模型部分还没有实现引起的。
```
ValueError: ('Expected `model` argument to be a `Model` instance, got ', None)
```

