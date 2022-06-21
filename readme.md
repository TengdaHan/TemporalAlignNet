# Temporal Alignment Networks for long-term Video 

Tengda Han, Weidi Xie, Andrew Zisserman. CVPR2022 Oral.

[[project page]](https://www.robots.ox.ac.uk/~vgg/research/tan/)
[[PDF]](https://www.robots.ox.ac.uk/~vgg/publications/2022/Han22a/han22a.pdf)
[[Arxiv]](https://arxiv.org/abs/2204.02968)
[[Video]](https://youtu.be/77dcM9CyHCY)

<img src="TAN_teaser.png" width="800">

### TLDR

* Natural instructional videos (e.g. from YouTube) has the visual-textual alignment problem, that introduces lots of noise and makes them hard to learn.
* Our model learns to predict:
  1. if the ASR sentence is alignable with the video,
  2. if yes, the most corresponding video timestamps.
* Our model is trained without human annotation, and can be used to clean-up the 
noisy instructional videos (we release an Auto-Aligned HTM dataset, **HTM-AA**). 
* In our paper, we show the auto-aligned HTM dataset is effective that it can improve the visual representation quality.

### Datasets
* [**HTM-Align**](htm_align/): A manually annotated 80-video subset for alignment evaluation.
* [**HTM-AA**](htm_aa/): A large-scale video-text paired dataset automatically aligned using our TAN without using any manual annotations.

### Tool
* [**Sentencify-text**](sentencify_text/): A pipeline to pre-process ASR text segments and get full sentences.

### Training TAN

* See instructions in [[train]](./train/)
### Using output of TAN for end-to-end training.

* See instructions in [[end2end]](./end2end/)

### Model Zoo
* We aim to release a model zoo for both TAN variants and end-to-end visual representations in July. Thanks for your interest!

### Reference
```
@InProceedings{han2022align,
  title={Temporal Alignment Network for long-term Video},  
  author={Tengda Han and Weidi Xie and Andrew Zisserman},  
  booktitle={CVPR},  
  year={2022}}
```




