# Rank & Sort Loss for Object Detection and Instance Segmentation - SOLOv2 Implementation

This repository provides an implementation of Rank & Sort Loss on SOLOv2. The implementation is based on [mmdetection v1](https://github.com/open-mmlab/mmdetection) and [this Solov2 implementation](https://github.com/WXinlong/SOLO). We plan to deprecate this repository after incorporating SOLOv2 into the [official Rank & Sort Loss Repository](https://github.com/kemaloksuz/RankSortLoss), which is based on mmdetection v2.

> [**Rank & Sort Loss for Object Detection and Instance Segmentation**](https://arxiv.org/abs/2107.11669),            
> [Kemal Oksuz](https://kemaloksuz.github.io/), Baris Can Cam, [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/), [Sinan Kalkan](http://www.kovan.ceng.metu.edu.tr/~sinan/),
> *ICCV 2021 (Oral Presentation). ([arXiv pre-print](https://arxiv.org/abs/2107.11669))*

## How to Cite

Please cite the paper if you benefit from our paper or the repository:
```
@inproceedings{RSLoss,
       title = {Rank & Sort Loss for Object Detection and Instance Segmentation},
       author = {Kemal Oksuz and Baris Can Cam and Emre Akbas and Sinan Kalkan},
       booktitle = {International Conference on Computer Vision (ICCV)},
       year = {2021}
}
```

## Installation
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.0.0). Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Models

| Backbone    |  Epoch  | mask AP | mask oLRP  |  Log  | Config | Model |
| :---------: | :-----: | :------------: | :------------: | :-------: | :-------: | :-------: |
|   ResNet-34 | 36 |   32.6  |   72.7  | Coming soon | [config](configs/ranksort_loss/ranksort_solov2_light_448_r34_fpn_3x.py) | Coming soon|
|  ResNet-101 | 36 |   39.7  |   66.9  | Coming soon | [config](configs/ranksort_loss/ranksort_solov2_r101_fpn_3x.py) |Coming soon|


## Running the Code

### Training Code
The configuration files of all models listed above can be found in the `configs/ranksort_loss` folder. As an example, to train Faster R-CNN with our RS Loss on 4 GPUs as we did, use the following command:

```
./tools/dist_train.sh configs/ranksort_loss/ranksort_solov2_light_448_r34_fpn_3x.py  4
```

### Test Code
The configuration files of all models listed above can be found in the `configs/ranksort_loss` folder. As an example:

```
./tools/dist_test.sh ranksort_loss/ranksort_solov2_light_448_r34_fpn_3x.py ${CHECKPOINT_FILE}  4  --show --out results_solo.pkl --eval segm
```