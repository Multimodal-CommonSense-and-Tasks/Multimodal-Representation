# SAAL: Sharpness-Aware Active Learning (ICML 2023) <br><sub>Official PyTorch implementation of the Discriminator Guidance </sub>
**[[Yoon-Yeong Kim]\*, [Youngjae Cho]\*, Joonho Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, Il-Chul Moon **   
<sup> * Equal contribution </sup> <br>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name Adam --isInit True
python3 main.py --data Cifar10 --acqMode Max_Diversity --optimizer_name SAM --isInit False
```

## Evaluation

- Data will be downloaded to folder 'data'.
- Result will be recorded to folder 'Results'.

## Results

Our model achieves the following performance on active learning settings:

| Optimizer  | FashionMNIST  |      SVHN     |    CIFAR-10   |   CIFAR-100   |
| ----------- |-------------- | ------------- | ------------- | ------------- |
|    Adam     |     85.8%    |     76.8%    |     54.4%    |     47.6%    |
|    SAM     |     86.3%    |     78.8%    |     57.0%    |     48.4%    |


## Contribution

We propose a new active learning method named Sharpness-Aware Active Learning, or SAAL.The proposed method considers theloss sharpness of data instances, which is strongly related to the generalization performance of deep learning.


## Reference


@InProceedings{pmlr-v202-kim23c,
  title = 	 {{SAAL}: Sharpness-Aware Active Learning},
  author =       {Kim, Yoon-Yeong and Cho, Youngjae and Jang, Joonho and Na, Byeonghu and Kim, Yeongmin and Song, Kyungwoo and Kang, Wanmo and Moon, Il-Chul},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {16424--16440},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/kim23c/kim23c.pdf},
  url = 	 {https://proceedings.mlr.press/v202/kim23c.html},
  abstract = 	 {While deep neural networks play significant roles in many research areas, they are also prone to overfitting problems under limited data instances. To overcome overfitting, this paper introduces the first active learning method to incorporate the sharpness of loss space into the acquisition function. Specifically, our proposed method, Sharpness-Aware Active Learning (SAAL), constructs its acquisition function by selecting unlabeled instances whose perturbed loss becomes maximum. Unlike the Sharpness-Aware learning with fully-labeled datasets, we design a pseudo-labeling mechanism to anticipate the perturbed loss w.r.t. the ground-truth label, which we provide the theoretical bound for the optimization. We conduct experiments on various benchmark datasets for vision-based tasks in image classification, object detection, and domain adaptive semantic segmentation. The experimental results confirm that SAAL outperforms the baselines by selecting instances that have the potentially maximal perturbation on the loss. The code is available at https://github.com/YoonyeongKim/SAAL.}
}


