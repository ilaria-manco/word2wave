# Word2Wave

Word2Wave is a simple method for text-controlled GAN audio generation. You can either follow the setup instructions below and use the source code and CLI provided in this repo or you can have a play around in the Colab notebook provided. Note that, in both cases, you will need to train a WaveGAN model first. You can also hear some examples [here](https://ilariamanco.com/word2wave/).


Colab playground [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c9DdSN_oiv0rcL9SH-W8-jfhcQf6iVYy?usp=sharing)

## Setup

First, clone the repository
```clone
git clone https://www.github.com/ilaria-manco/word2wave
```

Create a virtual environment and install the requirements:
```setup
cd word2wave
python3 -m venv /path/to/venv/
pip install -r requirements.txt
```

### WaveGAN generator
Word2Wave requires a pre-trained WaveGAN generator. In my experiments, I trained my own on the [Freesound Loop Dataset](https://zenodo.org/record/3967852#.YIlF931KhhE), using [this implementation](https://github.com/mostafaelaraby/wavegan-pytorch). To download the FSL dataset do:

```bash
$ wget https://zenodo.org/record/3967852/files/FSL10K.zip?download=1
```

and then train following the instructions in the WaveGAN repo. Once trained, place the model in the `wavegan` folder:

```
📂wavegan
  ┗ 📜gan_<name>.tar
```

### Pre-trained COALA encoders
You'll need to download the pre-trained weights for the COALA tag and audio encoders from the official [repo](https://github.com/xavierfav/coala). Note that the repo provides weights for the model trained with different configurations (e.g. different weights in the loss components). For more details on this, you can refer to the original code and paper. To download the model weights, you can run the following commands (or the equivalent for the desired model configuration)

```bash
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/saved_models/dual_ae_c/audio_encoder_epoch_200.pt
$ wget https://raw.githubusercontent.com/xavierfav/coala/master/saved_models/dual_ae_c/tag_encoder_epoch_200.pt
```

Once downloaded, place them in the `coala/models` folder:
```
📂coala
 ┣ 📂models
   ┣ 📂dual_ae_c
     ┣ 📜audio_encoder_epoch_200.pt
     ┗ 📜tag_encoder_epoch_200.pt
```

## How to use
For text-to-audio generation using the default parameters, simply do

```
$ python main.py "text prompt" --wavegan_path <path/to/wavegan/model> --output_dir <path/to/output/directory>
```

## Citations
Some of the code in this repo is adapted from the official [COALA repo](https://github.com/xavierfav/coala) and @mostafaelaraby's [PyTorch implenentation](https://github.com/mostafaelaraby/wavegan-pytorch) of the WaveGAN model. 

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
