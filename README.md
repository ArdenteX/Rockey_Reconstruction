<a name="readme-top"></a>

<div align="center">

<img height="60" src="https://github.com/VectorZhao/deepexo/blob/master/docs/icon-512%402x.png">
<img height="60" src="https://github.com/VectorZhao/deepexo/blob/master/docs/icon-512%402x-colorful.png">

<h1 align="center">DeepExo</h1>

English / [简体中文](./README_CN.md)

<b> One-Click to unlock the mysteries of exoplanet interiors with deepexo </b>, 

an integrated machine learning platform for planetary science.


</div>

## 👋🏻 Overview
Mixture density network using pytorch. Compared with the other implementation of MDN, this project using L2 and a couple of technique
instead of drop out layer to prevent the over fitting, this make the network more stably during training. This project is based on my tutor's [Rocky_Exoplanets_v2](https://github.com/VectorZhao/Rocky_Exoplanets_v2)
## ✨ Features
- 🌐 Unified Project Collection: We have meticulously integrated previously published machine learning models designed for predicting the internal structures of [gas giants](https://github.com/VectorZhao/ExtrasolarGasGiants) and [rocky exoplanets](https://github.com/VectorZhao/Rocky_Exoplanets_v2).
- 🔧 Cutting-edge Codebase: Our project has undergone a complete transformation with a redesigned code architecture implemented in PyTorch. This revamp includes the reimagining of essential components like the MDN layer, resulting in a more efficient and scalable framework.
- 📈 Superior Predictive Performance: Leveraging the power of our revamped codebase and retraining the MDN model with existing data, we've achieved substantial enhancements in predictive capabilities. Our model consistently demonstrates superior performance compared to its predecessors, delivering more accurate and reliable results for planetary science research.
- 🧩 Encapsulated Functions: To facilitate the scientific community, we've developed user-friendly encapsulated functions. These functions empower researchers to seamlessly initiate and train their own machine learning models for investigating the interiors of exoplanets. This abstraction simplifies the complexity of model development, accelerating scientific exploration.

## 🚀 Get Started

## Usage

Please visit [tortreinador](https://github.com/ArdenteX/tortreinador)

## Structure
1. This project using Elu in sigma layer for deal with the gradient disappear or explosion, at the same time, it is more
robust to noise than the tradition activate function 
    ```python
        self.root_layer = nn.Sequential(
            nn.Linear(self.i_s, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU()
        ).double()
        
        self.pi = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.n_g)
        ).double()
        
        self.mu = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.o_s * self.n_g)
        ).double()
        
        self.sigma = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.ELU(),
            nn.Linear(self.n_h, self.o_s * self.n_g)
        ).double()
    ```
2. This project apply L2 normalization and a bit of technique such as xavier init, warmup to prevent the over fitting.
    ```python
        # Xavier Init
        init_weights(model)
        model = nn.DataParallel(model)
        model.to(self.device)\
        
        # L2
        optimizer = torch.optim.Adam(split_weights(model), lr=self.lr, weight_decay=self.w_d)
        
        # Warmup
        warmup = WarmUpLR(optimizer, len(t_l) * 5)
    ```

## Performance
|                 | Hidden Size | Number of Gaussian | Batch Size | Learning Rate | NLL Loss     | R2         | Mse        | Speed (Epoch)    |
|:----------------|:------------|:-------------------|:-----------|:--------------|:-------------|:-----------|:-----------|:-----------------|
| Without Dropout | 256         | 10                 | 1024       | 0.0001984     | **-33.7150** | **0.9950** | **0.0002** | **79**           |
| Dropout         | 512         | 20                 | 512        | 0.0001        | -25.1895     | 0.9929     | 0.0003     | 120              |


- Speed(Epoch): Because the final performance of Without Dropout MDN is better than the other one, recording the epoch when each model's R2 is 0.9929 can compare the training speed between MDN of Without Dropout and Dropout

Negative Likelihood Loss Function (Dropout)
![img](Rock/Imgs/MDN_MRCk2_loss_20230524.png)

Negative Likelihood Loss Function (Without Dropout)
![img](Rock/Imgs/MRCk2_MDN20231129_TrainValLoss.png)

Probability density distribution heat map(Dropout)
![img](Rock/Imgs/img_2.png)

Probability density distribution heat map(Without Dropout)
![img](Rock/Imgs/prediction_MRCk2_20231201.png)

   

## Example
This project's train part is based on the module [tortreinador](https://github.com/ArdenteX/tortreinador).

```python
from tortreinador import train
from tortreinador.models.MDN import mdn, Mixture, NLLLoss
import torch

input_parameters = [
    'Mass',
    'Radius',
    'Fe_(Mg+Si)',
    'k2',
]


output_parameters = [
    'WRF',
    'MRF',
    'CRF',
    'CMF',
    'PRS_CMB',
    'TEP_CMB',
]

# df_all is your dataset

trainer = train.TorchTrainer()
# Model
model = mdn(len(input_parameters), len(output_parameters), 10, 256)

# Loss
criterion = NLLLoss()
pdf = Mixture()

# Optimizer
optim = torch.optim.Adam(trainer.xavier_init(model), lr=0.0001984, weight_decay=0.001)

# You can specify the input/output parameter and your dataset(Only support Dataframe currently) 
t_loader, v_loader, test_x, test_y, s_x, s_y = trainer.load_data(data=df_all, input_parameters=input_parameters,
                                                                 output_parameters=output_parameters,
                                                                 if_normal=True, if_shuffle=True)

# The default optimizer is Adam
t_l, v_l, val_r2, train_r2, mse = trainer.fit_for_MDN(
    t_loader, v_loader, criterion, model=model, mixture=pdf,
    model_save_path='D:\\Resource\\MDN\\', optim=optim, best_r2=0.5)
```

## 📚 References
- [Machine learning techniques in studies of the interior structure of rocky exoplanets](https://www.aanda.org/articles/aa/abs/2021/06/aa40375-21/aa40375-21.html)
- [Understanding the interior structure of gaseous giant exoplanets with machine learning techniques](https://www.aanda.org/articles/aa/abs/2022/02/aa42874-21/aa42874-21.html)
- [Machine-learning Inferences of the Interior Structure of Rocky Exoplanets from Bulk Observational Constraints](https://iopscience.iop.org/article/10.3847/1538-4365/acf31a)
                                                      



