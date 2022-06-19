# HTM-Align Dataset [[website]](https://www.robots.ox.ac.uk/~vgg/research/tan/#htm-align)

**HTM-Align** is a *manually annotated* 80-video subset of HowTo100M (HTM) dataset, to evaluate the alignment performance. It is a test set randomly sampled from `Food & Entertaining` category of HTM. These videos are not used for any training in our project.

### Download

* **HTM-Align**(616KB json) [[from Oxford server]](http://www.robots.ox.ac.uk/~htd/tan/htm_align.json)

### How To Load

```python
import json
htm_align = json.load(open('htm_align.json'))

print(len(htm_align))  # 80

print(htm_align['-3CEg4y7mQM'][1])
# [1, 10.739, 17.535, 'add extra virgin olive oil and garlic']
# format: [alignability (1/0), start(second), end(second), text]
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