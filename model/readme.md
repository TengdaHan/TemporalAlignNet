# Model directory

This directory contains the model code for Temporal Alignment Networks
and the model code for backbone S3D-word2vec.

### Preparation for [end-to-end training](../end2end/)

If you want to run experiments on S3D, 
you need to download the pretrained weight for S3D from [MIL-NCE paper](https://github.com/antoine77340/S3D_HowTo100M#getting-the-data).

```bash
cd s3d_milnce/
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
``` 

So your directory should be like:
```
model/
    s3d_milnce/
        __init__.py
        s3d_dict.npy
        s3d_howto100m.pth
        s3dg.py
    ...
```