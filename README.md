# Covid monai segmentation

Synthesis of Covid19 lesions using superpixels and neural cellular automata (nCA).

## Description

This work is in-progress.
* We use superpixels and nCA to produce early versions of the lesions.
* We evaluate the synthetic data used as an augmented dataset of the COVID-19 Lung CT Lesion Segmentation Challenge - 2020. (https://covid-segmentation.grand-challenge.org/)

## Getting Started

### Dependencies

MONAI, Pytorch and recent version of scikit-image.
```
pip install -q "git+https://github.com/Project-MONAI/MONAI#egg=monai[nibabel,ignite,tqdm]"
pip install -Uq scikit-image
```

### Executing program

* To run superpixels and nCA run on one slice
```
python3 segment_and_augment.py --SCAN_NAME volume-covid19-A-0014  --only_one_slice 34 --ITER_INNER 60
```
* To run segmentation pipeline
```
#to run baseline
python3 run_net.py train --data_folder "COVID-19-20_v2/Train"
#to run with augmented dataset
python3 run_net2.py train --data_folder "COVID-19-20_v2/Train"
```
### References

* [Mordvintsev A. et al. Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
* nCA code based on https://github.com/Mayukhdeb/differentiable-morphogenesis

## Local development
/content/drive/MyDrive/repositories/covid19_monai_segmentation

## Authors

Octavio Martinez Manzanera  
[@ocmtzman](https://twitter.com/ocmtzman)