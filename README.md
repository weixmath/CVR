# Pytorch code for CVR [ICCV2021].

Xin Wei*, Yifei Gong*, Fudong Wang, Xing Sun, Jian Sun. **Learning Canonical View Representation for 3D Shape Recognition with Arbitrary Views**. ICCV, accepted, 2021. [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wei_Learning_Canonical_View_Representation_for_3D_Shape_Recognition_With_Arbitrary_ICCV_2021_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wei_Learning_Canonical_View_ICCV_2021_supplemental.pdf)

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wei_2021_ICCV,
    author    = {Wei, Xin and Gong, Yifei and Wang, Fudong and Sun, Xing and Sun, Jian},
    title     = {Learning Canonical View Representation for 3D Shape Recognition With Arbitrary Views},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {407-416}
}
```

## Training

### Requiement

This code is tested on Python 3.6 and Pytorch 1.8 + 

### Dataset

First download the arbitrary views ModelNet40 dataset and put it under `data`

``

%### Command for training:

%`python train.py -name view-gcn -num_models 0 -weight_decay 0.001 -num_views 20 -cnn_name resnet18`

%The code is heavily borrowed from [[mvcnn-new]](https://github.com/jongchyisu/mvcnn_pytorch).

%We also provide a [trained view-GCN network](https://drive.google.com/file/d/1qkltpvabunsI7frVRSEC9lP2xDP6cDj3/view?usp=sharing) achieving 97.6% accuracy on ModelNet40.

%`https://drive.google.com/file/d/1qkltpvabunsI7frVRSEC9lP2xDP6cDj3/view?usp=sharing`
