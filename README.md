# README

## Installation

Install the required packages with the command below

bash install_packages.sh

## Convert ViSA to MVTecAD format

The following code converts ViSA dataset to MVTecAD format. Specify the source path (where original ViSA datsaet is located), and target path (where to save the converted ViSA dataset) and the "split_csv/1cls.csv" directory in the code.

```
python convert_visa_to_mvtec_format.py

```

## Prepare noisy long-tailed dataset

The following code generates a noisy long-tailed dataset of MVTecAD and converted ViSA(converting visa like MVTecAD dataset format is required). Specify the arguments the arguments to acquire the noisy long-tailed dataset. If "random_tail" is False, "tail_classes" should be specified with the list of tail classes and the number of "tail_classes" should be equal to int(num_classes * "tail_ratio"). Below are a few examples

```
python generate_noisy_tailed_dataset.py --dataset mvtec --tail_type pareto --random_tail True
python generate_noisy_tailed_dataset.py --dataset mvtec --tail_type k4 --random_tail True

python generate_noisy_tailed_dataset.py --dataset mvtec --tail_type k4 --random_tail False --tail_classes cable capsule wood zipper bottle transistor grid screw hazelnut
python generate_noisy_tailed_dataset.py --dataset mvtec --tail_type pareto --random_tail False --tail_classes cable capsule wood zipper bottle transistor grid screw hazelnut

```

## Train/test

After generating the noisy long-tailed dataset, run the code to train model. The configuration file for training or testing should be saved in ./configs directory.

```
python main.py --dataset --mvtec --noisy_lt_dataset paretno_nr0.1_seed42 --config tailedcore_mvtec
```

## Model Structure

Refer the files

./src/coreset_model.py for the code of each models

./src/sampler.py for the code of each samplers

which are the core codes of our method.

## Acknowledgement

The code is based on the repository of [PatchCore](https://github.com/amazon-science/patchcore-inspection)
