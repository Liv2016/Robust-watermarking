# Adaptive Robust Watermarking Method Based on Deep Neural Networks



## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [tensorflow = 1.1.14](https://www.tensorflow.org/) .
- Download dependencies

~~~bash
# 1. 克隆环境
conda env create -f TF114.yaml

# 2. 安装依赖
pip install -r requirements.txt
~~~




## Get Started
- Run `python main.py` for training.

~~~bash
# Recommended training methods for embedding 100 bit messages
python main.py --exp_name wm100 --secret_len 100 --cover_h 400 --cover_w 400 --num_epochs 200 --batch_size 4 --lr .0001 --dataset_path /home/Dataset/train/mirflickr --loss_lpips_ratio 1.5 --loss_mse_ratio 2 --loss_secret_ratio 3.5 --GPU 0 --damping_end 0.2
~~~



## Dataset
- In this paper, we use the commonly used dataset Mirflickr, and ImageNet.

- For train on your own dataset, change the code in `main.py`:

    `line23:  --dataset_path = 'your dataset' ` 



##  Tensorboard

~~~bash
# Monitoring with tensorboard
cd code path
tensorboard --logdir ./logs --port 6006
~~~



## result

ACC:

![ACC](.\result\ACC.png)

PSNR:

![ACC](.\result\PSNR.png)



LOSS:

![ACC](.\result\loss.png)
