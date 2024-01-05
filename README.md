# Bayesian Tensor Toolbox (BayTT)

   <div align=center><img src="figs/logo2.png" width = "200" height = "200" alt="logo" /></div>

(the repo is still under construction, some link and statistic could be wrong, we will release the full code soon)

<br />
BayTT is an open-source library collecting state-of-art models and baselines for Bayesian Tensor decomposition.

We provide a neat code base to decompose a sparse tensor in probabilistic ways, which cover three mainstream tasks now: **Sparse Tensor Decomposition,Streaming Tensor Decomposition, Temporal Tensor Decomposition**
.  We will add more topics like **Functional Tensor Decomposition** in the future. 

For each task, we made the leader borad evaluated on several classical datasets. We also provide the dataset in the repo. 


## Leaderboard
**Note: We will keep updating this leaderboard.** If you have proposed advanced and awesome models, you can send us your paper/code link or raise a pull request. We will add them to this repo and update the leaderboard as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.

### Sparse Tensor Decomposition
| Model name | Movielens 10K                  | Movielens 1M                                     | ACC                                                   | DBLP                                     | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [NEST](https://arxiv.org/abs/2310.06625)  | [NEST](https://arxiv.org/abs/2310.06625)              | [NEST](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [POND](https://github.com/yuqinie98/PatchTST)    | [POND](https://github.com/yuqinie98/PatchTST)   | [POND](https://github.com/yuqinie98/PatchTST)   | [POND](https://github.com/yuqinie98/PatchTST)   |  [POND](https://github.com/yuqinie98/PatchTST) |
| 🥉 3rd             |  [SparseHGP](https://github.com/yuqinie98/PatchTST)      | [SparseHGP](https://github.com/yuqinie98/PatchTST)           | [SparseHGP](https://github.com/yuqinie98/PatchTST)      | [SparseHGP](https://github.com/yuqinie98/PatchTST) |


  - [x] **NEST** - Conor Tillinghast and Shandian Zhe, “Nonparametric Decomposition of Sparse Tensors”, The Thirty-eighth International Conference on Machine Learning (ICML)[[(ICML 2021)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **POND** - Conor Tillinghast, Shikai Fang, Kai Zheng, and Shandian Zhe, “Probabilistic Neural-Kernel Tensor Decomposition”, IEEE International Conference on Data Mining (ICDM) [[ICDM 2020]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **SparseHGP** - Conor Tillinghast, Zheng Wang, and Shandian Zhe, “Nonparametric Sparse Tensor Factorization with Hierarchical Gamma Processes”, The 39th International Conference on Machine Learning (ICML) [[ICML 2022]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)

### Streaming Tensor Decomposition


| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [BASS-Tucker](https://arxiv.org/abs/2310.06625)  | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [SBDT](https://github.com/yuqinie98/PatchTST)    | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   |  [SBDT](https://github.com/yuqinie98/PatchTST) |
| 🥉 3rd             |  [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST)           | [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST) |



  - [x] **BASS-Tucker** - Shikai Fang, Akil Narayan, Robert Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition ”, The 39 International Conference on Machine Learning  [[(ICML 2022)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **SBDT** - Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, and Shandian Zhe, “Streaming Bayesian Deep Tensor Factorization” [[ICML 2021]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **SFTL** - Streaming Factor Trajectory Learning for Temporal Tensor Decomposition [[NeurIPS 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)


### Temporal Tensor Decomposition

| Model name | Movie-lens                   | DBLP                                  | ACC                                                   | demo                                      | 
| ---------------- |---------------------------------------------------| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | 
| 🥇 1st         | [BASS-Tucker](https://arxiv.org/abs/2310.06625)  | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)              | [BASS-Tucker](https://arxiv.org/abs/2310.06625)           |
| 🥈 2nd               |   [SBDT](https://github.com/yuqinie98/PatchTST)    | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   | [SBDT](https://github.com/yuqinie98/PatchTST)   |  [SBDT](https://github.com/yuqinie98/PatchTST) |
| 🥉 3rd             |  [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST)           | [SFTL](https://github.com/yuqinie98/PatchTST)      | [SFTL](https://github.com/yuqinie98/PatchTST) |



  - [x] **BASS-Tucker** - Shikai Fang, Akil Narayan, Robert Kirby, and Shandian Zhe, “Bayesian Continuous-Time Tucker Decomposition ”, The 39 International Conference on Machine Learning  [[(ICML 2022)]](https://users.cs.utah.edu/~shikai/file/ICML2022-BCTT-fang) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py).
  - [x] **SBDT** - Shikai Fang, Zheng Wang, Zhimeng Pan, Ji Liu, and Shandian Zhe, “Streaming Bayesian Deep Tensor Factorization” [[ICML 2021]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py).
  - [x] **SFTL** - Streaming Factor Trajectory Learning for Temporal Tensor Decomposition [[NeurIPS 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)


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
