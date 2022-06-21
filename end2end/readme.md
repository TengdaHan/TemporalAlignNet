# End-to-end Representation Learning with Aligned Video-Text Data

The output of Temporal Alignment Networks is
an automatically-aligned video-text data: [**HTM-AA**](../htm_aa/).

In our paper ([PDF](https://www.robots.ox.ac.uk/~vgg/publications/2022/Han22a/han22a.pdf)) Table 4 and Appendix D, 
we show this aligned video-text data can be used to train the backbone representation and improve its quality.

Settings  | Backbone | UCF101 | HMDB51 | K400
--- | --- | --- | --- | --- 
reported by [1]       | S3D | 82.7   | 53.1   | -
reproduce of [1]      | S3D | 82.1   | 55.2   | 55.7
finetune with HTM-AA  | S3D | 83.2   | 56.7   | 56.2

In this directory we provide code about these end-to-end experiments.

* [1] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan
Laptev, Josef Sivic, and Andrew Zisserman. End-to-end
learning of visual representations from uncurated instructional videos. In Proc. CVPR, 2020.

### Dataset

End-to-end training requires original HowTo100M videos, which needs a huge disk space (HTM full set is about 25 TB).
In our paper we use **HTM-AA** for training. It is a subset of HTM full set including about 25% videos, which takes about 6TB disk space.

* When storing HTM videos on the disk (SSD), we randomly split videos into multiple directories, each group contains at most 20,000 videos. We recommend to keep a hierarchical file structure to avoid filesystem issues. The example file structure is like:
    ```
    video_part_1/
        youtubevid1.mp4
        youtubevid2.mp4
        ...
    video_part_n/
        youtubevidx.mp4
    ```
* Then we prepare a `vid_to_path.json` file, it contains a dictionary that maps `vid` to video path, e.g.
    ```python
    {'youtubevid1': 'video_part_1/youtubevid1.mp4',
     'youtubevid2': 'video_part_1/youtubevid2.mp4',
     ...
     'youtubevidx': 'video_part_n/youtubevidx.mp4'}
    ```
* If you plan to do end-to-end training on HowTo100M, you have to download some original videos and make this `vid_to_path.json` file for your directory. Put this json file in this directory: `end2end/vid_to_path.json`.

* Also download the [HTM-AA](../htm_aa/) annotation file in this directory: `end2end/htm_aa_v1.csv`.

### End-to-end Training

Please refer to `main_nce.py` file for details.

Example command:
```sh
CUDA_VISIBLE_DEVICES=0 python main_nce.py --freezeBN --sim cos --auto_align_tag htm_aa_v1 \
--epochs 40 --batch_size 16 --num_frames 16 --fps 5
```

### Model Zoo

In the paper we only have time to try S3D-word2vec backbone. 

We plan to release models for other visual and textual backbone 
trained with HTM-AA dataset in July.
