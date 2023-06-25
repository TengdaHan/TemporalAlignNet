# Training TAN

### Data preparation

* See [../data/](../data/) for instructions

### Training commands

* First-stage pre-training (train joint/dual encoders separately):
`python main.py --model init --dataset htm-370k --batch_size 128 --use_text_pos_enc 0 --epochs 20`

* Second-stage training (train with encoders' concensus): 
`python main.py --model cotrain --dataset htm-370k --batch_size 128 --use_text_pos_enc 0 --epochs 20 --pretrain {} --loss_threshold 0.5`