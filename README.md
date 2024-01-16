# Bayesian Tensor Toolbox (BayTT)

   <div align=center><img src="figs/logo2.png" width = "200" height = "200" alt="logo" /></div>

(the repo is still under construction, some link and statistic could be wrong, we will release the full code soon)

<br />
BayTT is an open-source library collecting state-of-art models and baselines for Bayesian Tensor decomposition.

We provide a neat code base to decompose a sparse tensor in probabilistic ways, which cover three mainstream tasks now: **Sparse Tensor Decomposition, Streaming Tensor Decomposition, Temporal Tensor Decomposition**
.  We will add more topics like **Functional Tensor Decomposition** in the future. 

For each task, we made the leader borad evaluated on several classical datasets. We also provide the dataset in the repo. 


## Leaderboard
**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.

### Sparse Tensor Decomposition
| Model name | Movielens 10K                  | Movielens 1M                                     | ACC                                                   | DBLP                                     | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [NEST](http://proceedings.mlr.press/v139/tillinghast21a/tillinghast21a.pdf)  | [NEST](http://proceedings.mlr.press/v139/tillinghast21a/tillinghast21a.pdf)   | [NEST](http://proceedings.mlr.press/v139/tillinghast21a/tillinghast21a.pdf)     | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)    | [POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)     | [POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)     | [POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)     |  [POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)   |
| 🥉 3rd             |  [SparseHGP](https://proceedings.mlr.press/v162/tillinghast22a.html)      | [SparseHGP](https://proceedings.mlr.press/v162/tillinghast22a.html)           | [SparseHGP](https://proceedings.mlr.press/v162/tillinghast22a.html)      | [SparseHGP](https://proceedings.mlr.press/v162/tillinghast22a.html) |


  - [x] **NEST** - “Nonparametric Decomposition of Sparse Tensors”, [[(ICML 2021)]](http://proceedings.mlr.press/v139/tillinghast21a/tillinghast21a.pdf) [[Code]](https://github.com/ctilling/NEST).
  - [x] **POND** -  “Probabilistic Neural-Kernel Tensor Decomposition”, [[ICDM 2020]]([POND](https://users.cs.utah.edu/~zhe/pdf/ICDM_NKTF.pdf)  ) [[Code]](https://github.com/ctilling/POND).
  - [x] **SparseHGP** - “Nonparametric Sparse Tensor Factorization with Hierarchical Gamma Processes”,  [[ICML 2022]](https://proceedings.mlr.press/v162/tillinghast22a.html) [[Code]](https://github.com/ctilling/SparseTensorHGP)

### Streaming Tensor Decomposition


| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [SFTL](https://arxiv.org/abs/2310.06625)  | [SFTL](https://arxiv.org/abs/2310.06625)              | [SFTL](https://arxiv.org/abs/2310.06625)              | [SFTL](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [SBDT](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf)    | [SBDT](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf)   | [SBDT](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf)   | [SBDT](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf)   |  [SBDT](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf) |
| 🥉 3rd             |  [BASS-Tucker](https://github.com/yuqinie98/PatchTST)      | [BASS-Tucker](https://github.com/yuqinie98/PatchTST)           | [BASS-Tucker](https://github.com/yuqinie98/PatchTST)      | [BASS-Tucker](https://github.com/yuqinie98/PatchTST) |



  - [x] **BASS-Tucker** - Shikai Fang, Akil Narayan, Robert Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition ”, The 39 International Conference on Machine Learning  [[(ICML 2022)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **SBDT** - Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, and Shandian Zhe, “Streaming Bayesian Deep Tensor Factorization” [[ICML 2021]](http://proceedings.mlr.press/v139/fang21d/fang21d.pdf) [[Code]](https://github.com/xuangu-fang/Streaming-Bayesian-Deep-Tensor).
  - [x] **SFTL** - Streaming Factor Trajectory Learning for Temporal Tensor Decomposition [[NeurIPS 2023]](https://openreview.net/pdf?id=Qu6Ln7d9df) [[Code]](https://github.com/xuangu-fang/Streaming-Factor-Trajectory-Learning)


### Temporal Tensor Decomposition

| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [SFTL](https://arxiv.org/abs/2310.06625)  | [SFTL](https://arxiv.org/abs/2310.06625)              |[DEMOTE](https://proceedings.mlr.press/v162/wang22ar.html)       | [SFTL](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [NON-FAT](https://github.com/yuqinie98/PatchTST)    | [DEMOTE](https://proceedings.mlr.press/v162/wang22ar.html)  | [NON-FAT](https://proceedings.mlr.press/v162/wang22ar.html)   | [DEMOTE](https://arxiv.org/pdf/2310.19666.pdf)  |  
| 🥉 3rd             |  [BCTT](https://proceedings.mlr.press/v162/fang22b.html)      | [BCTT](https://proceedings.mlr.press/v162/fang22b.html)           | [NON-FAT](https://proceedings.mlr.press/v162/wang22ar.html)      | [NON-FAT](https://proceedings.mlr.press/v162/wang22ar.html) |



  - [x] **DEMOTE** - Zheng Wang, Shikai Fang, Shibo Li, and Shandian Zhe, “Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes”  [[(NeurIPS 2023)]](https://arxiv.org/pdf/2310.19666.pdf) [[Code]](https://github.com/wzhut/Dynamic-Tensor-Decomposition-via-Neural-Diffusion-Reaction-Processes).
  - [x] **NON-FAT** - Zheng Wang, and Shandian Zhe, “Nonparametric Factor Trajectory Learning for Dynamic Tensor Decomposition”[[ICML 2022]](https://proceedings.mlr.press/v162/wang22ar.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **BCTT** - Shikai Fang, Akil Narayan, Robert M. Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition”[[ICML 2022]](https://proceedings.mlr.press/v162/fang22b.html) [[Code]](https://github.com/xuangu-fang/Bayesian-Continuous-Time-Tucker-Decomposition)


## List of Byesian Tensor Models in this Repo

### Bayesian Sparse Tensor Decomposition
| Name | Description                     |                                 |                                               |                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SparseHGP      | Sparse Tensor Factorization with Hierarchical Gamma Processes  | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| NEST       | Non-linear decompostion based on Dirichlet processes and Gaussian processes  | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| POND       | Non-linear decompostion based on Deep Kernel Gaussian Process  | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| GPTF       | Non-linear decompostion based on Sparse Gaussian Process| [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  SVI-CP               |  Bayesian CP decompostion with stochastic variational inference(SVI) | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  SVI-Tucker         | Bayesian Tucker decompostion with stochastic variational inference(SVI)    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  CEP-CP         | Bayesian CP decompostion with conditional expectation propagation(CEP)      | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| CEP-Tucker         |  Bayesian Tucker decompostion with conditional expectation propagation(CEP)   | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 



### Bayesian Streaming Tensor Decomposition
| Name | Description                     |                                 |                                               |                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SNBDT     | Streaming Nonlinear decomposition with random Fourier features | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  SBDT      | Streaming Deep decompostion with sparse BNN| [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  BASS-Tucker          |  Streaming Tucker decompostion with sparse Tucker core | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  POST         | Streaming CP decompostion with SVB update    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  ADF-CP         | Streaming CP decompostion with ADF update    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| ADF-Tucker         |  Streaming Tucker decompostion with ADF update    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 



### Bayesian Temporal Tensor Decomposition
| Name | Description                     |                                 |                                               |                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SFTL     | Streaming Temporal CP/Tucker  with time-varing latent factors| [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| DEMOTE    |  Temporal tensor as Diffusion-Reaction Processes on Graph| [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  BCTT      | Temporal Tucker decompostion with time-varing tucker core| [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|   NON-FAT        |  GP priors + Fourier Transform   | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  THIS-ODE         |   Temporal Tensor decompostion with neuralODE   | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
|  CT-GPTF        | Streaming CP decompostion with ADF update    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 
| CT-CP         |  Streaming Tucker decompostion with ADF update    | [demo](https://arxiv.org/abs/2310.06625)              | [paper](https://arxiv.org/abs/2310.06625)             | [origin code](https://arxiv.org/abs/2310.06625)          | 


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
