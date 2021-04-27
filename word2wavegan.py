import os
import json
import pickle
import logging
from urllib.request import urlretrieve

import torch
from torch import nn

from wavegan import WaveGANGenerator
from coala import TagEncoder, AudioEncoder
from audio_prepro import preprocess_audio


class Word2WaveGAN(nn.Module):
    def __init__(self, config):
        super(Word2WaveGAN, self).__init__()
        # self.device = config.device
        self.device = "cuda"
        self.pretrained_model_path = config.pretrained_model_path

        self.load_wavegan()
        self.load_coala()
        self.init_latents()

    def load_wavegan(self, slice_len=16384, model_size=32):
        path_to_model = os.path.join(self.pretrained_model_path)
        self.generator = WaveGANGenerator(slice_len=slice_len, model_size=model_size, use_batch_norm=False,num_channels=1)
        checkpoint = torch.load(path_to_model, map_location="cuda")
        self.generator.load_state_dict(checkpoint['generator'])

    def load_coala(self, model_name="dual_e_c"): 
        coala_path = os.path.join("coala", model_name)
        tag_encoder_url = "https://github.com/xavierfav/coala/blob/master/saved_models/{}/tag_encoder_epoch_200.pt".format(model_name)
        audio_encoder_url = "https://github.com/xavierfav/coala/blob/master/saved_models/{}/audio_encoder_epoch_200.pt".format(model_name)
        tag_encoder_path = os.path.join(coala_path, os.path.basename(tag_encoder_url))
        audio_encoder_path = os.path.join(coala_path, os.path.basename(audio_encoder_url))
        # TODO below does not work due to corrupted download - download manually instead
        if not os.path.exists(coala_path):
            os.mkdir(coala_path)
            logging.info("Downloading COALA model weights from {}".format(audio_encoder_url))
            urlretrieve(tag_encoder_url, tag_encoder_path)
            urlretrieve(audio_encoder_url, audio_encoder_path)

        self.tag_encoder = TagEncoder()
        self.tag_encoder.load_state_dict(torch.load(tag_encoder_path))
        # tag_model.to("cuda")
        self.tag_encoder.eval()

        self.audio_encoder = AudioEncoder()
        self.audio_encoder.load_state_dict(torch.load(audio_encoder_path))
        # audio_model.to("cuda")
        self.audio_encoder.eval()

    def init_latents(self, size=1, latent_dim=100):
        noise = torch.FloatTensor(size, latent_dim)
        noise.data.normal_()
        # latents = torch.nn.Parameter(noise, requires_grad=True)
        self.latents = torch.nn.Parameter(noise)

    def encode_text(self, text_prompt):
        id2tag = json.load(open('coala/id2token_top_1000.json', 'rb'))
        tag2id = {tag: id for id, tag in id2tag.items()}

        sentence_embedding = torch.zeros(1152).cuda()
        for word in text_prompt.split(" "):
            tag_vector = torch.zeros((1, 1000)).cuda()
            tag_vector[0, int(tag2id[word])] = 1
            embedding, embedding_d = self.tag_encoder(tag_vector)
            sentence_embedding += embedding_d.squeeze(0)
        return sentence_embedding
    
    def encode_audio(self, audio):
        x = preprocess_audio(audio).to("cuda")
        scaler = pickle.load(open('coala/scaler_top_1000.pkl', 'rb'))
        x *= torch.tensor(scaler.scale_).cuda()
        x += torch.tensor(scaler.min_).cuda()
        x = torch.clamp(x, scaler.feature_range[0], scaler.feature_range[1])
        embedding, embedding_d = self.audio_encoder(x.unsqueeze(0).unsqueeze(0))
        return embedding_d

    def latent_space_interpolation(self, latents=None, n_samples=1):
        if latents is None:
            z_test = sample_noise(2)
        else:
            z_test = latents
        interpolates = []
        for alpha in np.linspace(0, 1, n_samples):
            interpolate_vec = alpha * z_test[0] + ((1 - alpha) * z_test[1])
            interpolates.append(interpolate_vec)
        interpolates = torch.stack(interpolates)
        generated_audio = self.generator(interpolates)
        return generated_audio

    def synthesise_audio(self, noise):
        generated_audio = self.generator(noise).view(-1)
        return generated_audio

    def coala_loss(self, audio, text):
        # logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # logit_scale = logit_scale.exp()

        text_embedding = self.encode_text(text)
        audio_embedding = self.encode_audio(audio)

        text_embedding = text_embedding / text_embedding.norm()
        audio_embedding = audio_embedding / audio_embedding.norm()
        
        # cos_sim = logit_scale * audio_embedding @ text_embedding.t()
        # cos_sim = 1/cos_sim * 100
        cos_dist =  (1 - audio_embedding @ text_embedding.t()) / 2

        return cos_dist

    def forward(self, text):
        audio = self.generator(self.latents).view(-1)
        loss = self.coala_loss(audio, text)
        return audio, loss
