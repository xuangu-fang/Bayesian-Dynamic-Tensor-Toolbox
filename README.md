# Bayesian Dynamic Tensor Toolbox (BayDTT)


  ![logo](figs/logo2.png =0.8)


BayDTT is an open-source library collecting state-of-art models and baselines for Bayesian Dynamic Tensor decomposition.

We provide a neat code base to decompose a sparse tensor in probabilistic and dynamic ways, which cover two mainstream tasks now: **Streaming Tensor Decomposition, Temporal Tensor Decomposition**
. Some methods of standard ** Static Tensor Decomposition** under the Bayesian framework are also included. We will add more topics like **Functional Tensor Decomposition** in the future. 

For each task, we made the leader borad evaluated on several classical datasets. We also provide the dataset in the repo. 

## Streaming Tensor Decomposition


| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [BASS-Tucker](https://arxiv.org/abs/2310.06625)  | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [SBDT](https://github.com/yuqinie98/PatchTST)    | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   |  [SBDT](https://github.com/yuqinie98/PatchTST) |
| 🥉 3rd             |  [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST)           | [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST) |


**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **BASS-Tucker** - Shikai Fang, Akil Narayan, Robert Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition ”, The 39 International Conference on Machine Learning  [[(ICML 2022)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **SBDT** - Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, and Shandian Zhe, “Streaming Bayesian Deep Tensor Factorization” [[ICML 2021]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **SFTL** - Streaming Factor Trajectory Learning for Temporal Tensor Decomposition [[NeurIPS 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)


## Temporal Tensor Decomposition

| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [BASS-Tucker](https://arxiv.org/abs/2310.06625)  | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [SBDT](https://github.com/yuqinie98/PatchTST)    | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   |  [SBDT](https://github.com/yuqinie98/PatchTST) |
| 🥉 3rd             |  [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST)           | [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST) |


**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.
  - [x] **BASS-Tucker** - Shikai Fang, Akil Narayan, Robert Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition ”, The 39 International Conference on Machine Learning  [[(ICML 2022)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **SBDT** - Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, and Shandian Zhe, “Streaming Bayesian Deep Tensor Factorization” [[ICML 2021]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **SFTL** - Streaming Factor Trajectory Learning for Temporal Tensor Decomposition [[NeurIPS 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)

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
