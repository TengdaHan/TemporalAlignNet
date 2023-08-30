# HowTo100M Data Release

We share additional features/data of the HowTo100M dataset for future research. [What is HowTo100M?](https://www.di.ens.fr/willow/research/howto100m/)

### WhisperX
It is commonly known that the ASR from YouTube has some noises, including synchronization issue (the timestamp does not associated with the speech perfectly) and translation issue (failure to recognize language and then transcribe in EN).

We use the time-wise accurate [WhisperX](https://github.com/m-bain/whisperX) package to process all the HowTo100M audio files, which gives **word-level** timestamps and highly accurate language recognition. WhisperX is build on [OpenAI's Whisper](https://github.com/openai/whisper) with additional phoneme alignment module to ensure accurate timestamp of the ASR.

We used the `whisper-large-v2` version. The entire processing costs about 9000 USD on AWS.
For non-EN language, we provide ASR and word-level timestamps in the local langauge (if supported by whisperX), as well as English translation, but with sentence-level timestamps.

* Downloading script: [here (64 tar.gz files of json, totally 25GB)](whisperx/download_whisperx_script.sh)
* Extraction script: [here (after unzip about 130GB)](whisperx/extract_whisperx_script.sh)
* Language detection output: [download link](http://www.robots.ox.ac.uk/~htd/howto100m/language_detection.csv)
* For future reference: our [language detection script](whisperx/language_detect.py) and [whisperx script](whisperx/transcribe_or_translate.py)

### Visual Features

We provide recent/stronger visual features for HowTo100M. Following [Miech et al.](https://www.di.ens.fr/willow/research/howto100m/), we provide features at **1 vector-per-second**. For the original S3D features, please refer to [Miech et al.](https://www.di.ens.fr/willow/research/howto100m/)

Currently we provide the following visual features:
* [InternVideo-MM-L14](https://github.com/OpenGVLab/InternVideo)
    * Downloading script: (Comming soon, 1.4TB)
    * Our [feature extraction script](visual/extract_feature_template.py)
* [CLIP-ViT-L-14](https://github.com/openai/CLIP)
    * Downloading script: (Comming soon, 700GB)
    * Our [feature extraction script](visual/extract_feature_template.py)

#### Feature quality benchmarked on HTM-Align
Without any (learnable) joint visual-language model, we measure the backbone visual-langauge feature quality on [HTM-Align](../htm_align/) -- which is similar to a retrieval setting.

| Model | Setting | Recall |
| --- | --- | --- |
| MILNCE | global | 0.287 |
| MILNCE | overlap-seq | 0.342 |
| CLIP ViT/B-32 | global | 0.175 |
| CLIP ViT/B-32 | overlap-seq | 0.234 |
| CLIP ViT/B-16 | global | 0.221 |
| CLIP ViT/B-16 | overlap-seq | 0.278 |
| CLIP ViT/L-14 | global | 0.256 |
| CLIP ViT/L-14 | overlap-seq | 0.309 |
| InternVideo-MM-L14 | global | 0.406 |
| InternVideo-MM-L14 | overlap-seq | 0.437 |


### Reference
If you find these data helpful, please consider citing us:
```bibtex
@InProceedings{han2022align,
  title={Temporal Alignment Network for long-term Video},  
  author={Tengda Han and Weidi Xie and Andrew Zisserman},  
  booktitle={CVPR},  
  year={2022}}
```
