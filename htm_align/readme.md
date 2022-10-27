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

### Performance on HTM-Align

<table>
<thead>
  <tr>
    <th>method</th>
    <th>time window for inference</th>
    <th>HTM-Align R@1</th>
    <th>HTM-Align ROC-AUC</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>CLIP ViT-B32</td>
    <td>global</td>
    <td>17.5</td>
    <td>70.9*</td>
  </tr>
  <tr>
    <td>CLIP ViT-B32</td>
    <td>64s moving window**</td>
    <td>23.4</td>
    <td>70.9*</td>
  </tr>
  <tr>
    <td>MIL-NCE</td>
    <td>global</td>
    <td>28.7</td>
    <td>73.3*</td>
  </tr>
  <tr>
    <td>MIL-NCE</td>
    <td>64s moving window**</td>
    <td>34.2</td>
    <td>73.4*</td>
  <tr>
    <td>TAN (HTM-370K) exp-D</td>
    <td>64s moving window</td>
    <td><b>49.8</b></td>
    <td><b>75.1</b></td>
  </tr>
  </tr>
</tbody>
</table>

*: since the model does not have a binary classifier for alignability, for each sentence, 
we first compute the sentence-visual similarity scores,
then take its maximum score over time as the alignability measurement to compute ROC-AUC.

**: in the paper, we only reported CLIP and MIL-NCE results with the 'global' time window setting, since CLIP and MIL-NCE do not use long-range temporal context. Here we also show their results with the 'moving window' setting for a fair comparison.

Note: After fixing a bug in ROC-AUC metric, 
the reproduced ROC-AUC scores are different with the numbers originally reported in the paper Table 1. 
Please consider comparing with the new results here.
We will update our arXiv paper for this correction.

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