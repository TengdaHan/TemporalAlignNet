# HTM-AA Dataset [[website]](https://www.robots.ox.ac.uk/~vgg/research/tan/#htm-aa)

**HTM-AA** means Auto-Aligned (AA) version of HowTo100M (HTM) dataset. It is an output of our Temporal Alignment Networks and a final goal of this project. 

HTM-AA is a large-scale **paired video-text** dataset, automatically obtained without any human annotation. 
In our paper Table 4, we show it can improve the backbone visual representation.

For a video from the HowTo100M dataset, HTM-AA provides: 
1. the visually alignable sentences taken from the YouTube ASR, 
2. their corresponding video timestamps (in second).

### Download

* **HTM-AA-v1**(329MB csv) [[from Oxford server]](http://www.robots.ox.ac.uk/~htd/tan/htm_aa_v1.csv)
[[from Google Drive]](https://drive.google.com/file/d/1_V2LfMil5wnWfxDkzBtSYCw7YUUSjzqE/view?usp=sharing)

### Statistics

* [[website]](https://www.robots.ox.ac.uk/~vgg/research/tan/htm_aa_stats.html)


### How To Load

```python
import pandas as pd
htm_aa = pd.read_csv('htm_aa_v1.csv')

print(htm_aa.iloc[42].to_dict())
# {'vid': '6yooogsTG8k',
#  'timestamp': 284,
#  'text': "and starting with pink i'm just going to knead that a little bit just to make it nice and smooth"}
```

### Reference

If you find this dataset useful for your project, please consider citing our paper:
```bibtex
@InProceedings{Han2022TAN,
    author       = "Tengda Han and Weidi Xie and Andrew Zisserman",
    title        = "Temporal Alignment Networks for Long-term Video",
    booktitle    = "CVPR",
    year         = "2022",
}
```