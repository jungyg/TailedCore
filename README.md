## 📢 **News and Updates**

- ✅ Mar 10, 2025. TailedCore **code** released.

<div align="center">

# [CVPR 2025] TailedCore : Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection

### This is the official repository for [TailedCore](https://arxiv.org/abs/2504.02775) (CVPR 2025).

Yoon Gyo Jung*, Jaewoo Park*, Jaeho Yoon*, Kuan-Chuan Peng,

Wonchul Kim, Andrew Beng Jin Teoh, Octavia Camps

*: Equal Contribution

[Project Page](https://jungyg.github.io/TailedCore_site)

</div>

<div align="center">

<img src="./figs/neu.png" height="100" alt="" align="center" style="margin-right: 10px;" />
<img src="./figs/aiv.png" height="100" alt="" align="center" style="margin-right: 10px;" />
<img src="./figs/yonsei.png" height="100" alt="" align="center" style="margin-right: 10px;" />
<img src="./figs/merl.png" height="100" alt="" align="center" style="margin-right: 10px;" />

</div>

TL;DR: We suggest a novel practical challenging anomaly detection task, noisy long-tailed anomaly detection where tail classes are unknown and head classes are contaminated. We suggest TailSampler, which first tail classes with class size estimation and denoise head classes seprately.

<div align="center">
  <img src="figs/bias.png" width="650px" height="300px">
</div>

**Performance comparison with baselines**

<div align="center">
  <img src="figs/method.png" width="800px" height="300px">
</div>

**Pipeine of TailedCore**

<div align="center">
  <img src="figs/dillema.png" width="650px" height="300px">
</div>

**Noise discriminative models remove tail classes(left) while greedy sampling samples both tail and noise**

<div align="center">
  <img src="figs/ablation_noise_ratio.png" width="650px" height="300px">
</div>

**Ablation with noise ratio comparing with baselines**



## 🪒 *Installation*

Install the required packages with the command below

bash install_packages.sh

## 💾 *Dataset Preparation*

### Noisy Long-Tail MVTecAD

Download MVTecAD dataset from [the link](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), and place it at, for example, `./datasets/mvtecad`. Then run the following
```
bash make_all_mvtecad_nlt.sh
```

### Noisy Long-Tail VisA
Download VisA dataset from [the link](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar), and place it at, for example, `./datasets/visa`. Then run the following
```
bash make_all_mvtecad_nlt.sh
```

## Train/test

After generating the noisy long-tailed dataset, run the code to train model. The configuration file for training or testing should be saved in `./configs` directory.

```
python main.py --dataset --mvtec --noisy_lt_dataset paretno_nr0.1_seed42 --config tailedcore_mvtec
```

## Code Structure

Refer the files
[`coreset_model`](./src/coreset_model.py) for the code of each models
[`sampler`](./src/sampler.py) for the code of each samplers

which are the core codes of our method.

## **Citations**

**The following is a BibTeX reference:**

```latex
@inproceedings{jung2025tailedcore,
  title={{TailedCore}: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection},
  author={Yoon Gyo Jung and Jaewoo Park and Jaeho Yoon and Kuan-Chuan Peng and Wonchul Kim and Andrew Beng Jin Teoh and Octavia Camps},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  url={https://arxiv.org/abs/2504.02775},
  year={2025}
}
```

## Acknowledgement

The code is based on the repository of [PatchCore](https://github.com/amazon-science/patchcore-inspection)
