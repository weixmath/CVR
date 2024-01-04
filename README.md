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

`https://drive.google.com/file/d/1RfE0aJ_IXNspVs610BkMgcWDP6FB0cYX/view?usp=sharing`

Link of arbitrary views ScanObjectNN dataset:

`https://drive.google.com/file/d/10xl-S8-XlaX5187Dkv91pX4JQ5DZxeM8/view?usp=sharing`

Aligned-ScanObjectNN dataset: `https://drive.google.com/file/d/1ihR6Fv88-6FOVUWdfHVMfDbUrx2eIPpR/view?usp=sharing`

Rotated-ScanObjectNN dataset: `https://drive.google.com/file/d/1GCwgrfbO_uO3Qh9UNPWRCuz2yr8UyRRT/view?usp=sharing`

#### The code is borrowed from [[OTK]](https://github.com/claying/OTK) and [[view-GCN]](https://github.com/weixmath/view-GCN).
