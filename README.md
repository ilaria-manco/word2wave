# Word2WaveGAN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) Colab playground

## Setup

First, clone the repository
```clone
git clone https://www.github.com/ilaria-manco/word2wavegan
```

Create a virtual environment and install the requirements:
```setup
cd word2wavegan
python3 -m venv /path/to/venv/
pip install -r requirements.txt
```

## WaveGAN Generator
Word2WaveGAN requires a pre-trained WaveGAN generator. In our experiments, we trained our own on the [Freesound Loop Dataset](https://zenodo.org/record/3967852#.YIlF931KhhE), using [this implementation](https://github.com/mostafaelaraby/wavegan-pytorch). 

## Pre-trained COALA encoders
You'll also need to obtain some files from the original [COALA repo](https://github.com/xavierfav/coala) and place them under the `coala` folder as shown below

```
📂coala
 ┣ 📂models
 ┣ 📜id2token_top_1000.json
 ┗ 📜scaler_top_1000.pkl
```

To download the files, run the following commands 

```bash
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/scaler_top_1000.pkl
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/scaler_top_1000.pkl
```

You'll also need to download the pre-trained weights for the COALA tag and audio encoders. Note that the COALA repo provides weights for the model trained with different configurations (e.g. different weights in the loss components). For more details on this, you can refer to the original code and paper. To download the model weights, you can run the following commands (or the equivalent for the desired model configuration)

```bash
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/saved_models/dual_ae_c/audio_encoder_epoch_200.pt
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/saved_models/dual_ae_c/tag_encoder_epoch_200.pt
```

## Citations
Some of the code in this repo is adapted from the original [COALA repo](https://github.com/xavierfav/coala) and @mostafaelaraby's [PyTorch implenentation](https://github.com/mostafaelaraby/wavegan-pytorch) of the WaveGAN model. 

```bibtex
@inproceedings{donahue2018adversarial,
  title={Adversarial Audio Synthesis},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

```bibtex
@article{favory2020coala,
  title={Coala: Co-aligned autoencoders for learning semantically enriched audio representations},
  author={Favory, Xavier and Drossos, Konstantinos and Virtanen, Tuomas and Serra, Xavier},
  journal={arXiv preprint arXiv:2006.08386},
  year={2020}
}
```
