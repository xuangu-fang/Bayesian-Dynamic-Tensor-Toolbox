# Bayesian Dynamic Tensor Toolbox (BayDTT)
BayDTT is an open-source library collecting state-of-art models and baselines for Bayesian Dynamic Tensor decomposition.

We provide a neat code base to decompose a sparse tensor in probabilistic and dynamic ways, which cover two mainstream tasks now: **Streaming Tensor Decomposition, Temporal Tensor Decomposition**
. Some methods of standard ** Static Tensor Decomposition** under the Bayesian framework are also included. We will add more topics like **Functional Tensor Decomposition** in the future. 

## Streaming Tensor Decomposition

## Temporal Tensor Decomposition

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtained the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Then place the downloaded data under the folder `./dataset`. Here is a summary of supported datasets.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{fang2022bayesian,
  title={Bayesian Continuous-Time Tucker Decomposition},
  author={Fang, Shikai and Narayan, Akil and Kirby, Robert and Zhe, Shandian},
  booktitle={International Conference on Machine Learning},
  pages={6235--6245},
  year={2022},
  organization={PMLR}
}
```
