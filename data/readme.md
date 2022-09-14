# Data directory for TAN experiments

The directory for data loaders for TAN experiments (not for [end-to-end](../end2end/) experiments).


### Instruction

We have shared the pre-processed ASR sentence of the entire HowTo100M dataset (i.e. the output of `sentencify-text` module for all HTM videos) on VGG server: https://www.robots.ox.ac.uk/~vgg/research/tan/index.html#htm-sentencify

You should download and place the files as

```sh
data/
    sentencified_htm_370k.json
    htm_align.json
```

### Preparation (optional)

1. Pre-process HowTo100M ASR text with [sentencify-text module](../sentencify_text/process_htm.py)

2. Store the processed ASR sentences as separate csv files, e.g. `abcdefghijk.csv` contains
    ```sh
    start,end,text
    4.13,6.50,"so we've moved location for our dessert"
    6.50,8.36,"and as you can see there's an amazing area"
    ...
    ```

3. Prepare an `vid_to_asr.json` file in this directory, containing a dictionary mapping `vid` to the csv path for ASR, e.g.
    ```python
    {'abcdefghijk': 'your_path/abcdefghijk.csv',
     ...}
    ```

4. Test your data preparation, run:
    ```sh
    python loader_htm.py
    ```
    You should see a list of strings without error.

