import argparse
import torch
import librosa
import os
import json
import logging
import numpy as np
import torchaudio

from word2wavegan import Word2WaveGAN

logging.basicConfig(level = logging.INFO)
import warnings
warnings.filterwarnings('ignore')

def play_my_words(text_prompt, args):
    word2wave = Word2WaveGAN(args)
    word2wave.cuda()

    for name, param in word2wave.named_parameters():
        # if name != "latents" and "generator" not in name:
        if name != "latents":
            param.requires_grad = False

    optimizer = torch.optim.Adam(
    params=[word2wave.latents],
    lr=args.lr,
    betas=(0.9, 0.999)
    )

    i = 0

    _, words_in_dict, words_not_in_dict = word2wave.tokenize_text(text_prompt)
    if not words_in_dict:
        raise Exception("All the words in the text prompt are out-of-vocabulary, please try with another prompt")
    elif words_not_in_dict:
        missing_words = ", ".join(words_not_in_dict)
        logging.info("Out-of-vocabulary words found, ignoring: \"{}\"".format(missing_words))
    logging.info("Making sounds to match the following text: {}".format(" ".join(words_in_dict)))
    
    while i < args.steps:
        audio, loss = word2wave(text_prompt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Step {i}', f'|| Loss: {loss.data.cpu().numpy()[0]}')
            # print(word2wave.latents)

        if loss <= args.threshold:    
            break
        
        i += 1
    
    audio_to_save = np.array(audio.detach().cpu().numpy())
    librosa.output.write_wav(os.path.join(args.output_dir, text_prompt + ".wav"), audio_to_save, args.sample_rate)

    if loss > args.threshold:
        logging.info("The optimisation failed to generate audio that is sufficiently similar to the given prompt. You may wish to try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_prompt", type=str, default="water", help="text prompt to guide the audio generation")
    parser.add_argument("--lr", type=float, default=0.04, help="learning rate")
    parser.add_argument("--steps", type=int, default=10000, help="number of optimization steps")
    parser.add_argument("--coala_model_name", type=str, default="dual_e_c", help="coala model name (can be one of [dual_e_c, dual_ae_c]")
    parser.add_argument("--wavegan_path", type=str, default="wavegan/gan_fs_loop_32.tar", help="path to the pretrained wavegan model")
    parser.add_argument("--threshold", type=float, default=0.15, help="threshold below which optimisation stops")
    parser.add_argument("--batch", type=bool, default=False, help="whether to run batch of experiments with all tags")
    parser.add_argument("--output_dir", type=str, default="output_new", help="path to store results")
    parser.add_argument("--sample_rate", type=int, default=16000)
    

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.batch:
        id2tag = json.load(open('coala/id2token_top_1000.json', 'rb'))
        for id, tag in id2tag.items():
            play_my_words(tag, args)
    
    else:
        play_my_words(args.text_prompt, args)
