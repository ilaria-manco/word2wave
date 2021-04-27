import argparse
import torch
import librosa
import os
import json
import numpy as np

from word2wavegan import Word2WaveGAN

def play_my_words(text_prompt, args):
    word2wave = Word2WaveGAN(args)
    word2wave.cuda()

    for name, param in word2wave.named_parameters():
        if name != "latents":
            param.requires_grad = False

    optimizer = torch.optim.Adam(
    params=[word2wave.latents],
    lr=args.lr,
    betas=(0.9, 0.999)
    )

    i = 0
    print(text_prompt)
    while i < args.steps:
        # audio, loss = word2wave(args.text_prompt)
        audio, loss = word2wave(text_prompt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 100 == 0:
        #     print(f'Step {i}', f'|| Loss: {loss.data.cpu().numpy()[0]}')
        #     # print(word2wave.latents)

        if loss < 0.11:
            audio_to_save = np.array(audio.detach().cpu().numpy())
            # librosa.output.write_wav(os.path.join(args.output_dir, args.text_prompt + ".wav"), audio_to_save, 16000)
            librosa.output.write_wav(os.path.join(args.output_dir, text_prompt + ".wav"), audio_to_save, 16000)    
            break
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("text_prompt", type=str, help="text prompt to guide the audio generation")
    parser.add_argument("--lr", type=float, default=0.04, help="learning rate")
    parser.add_argument("--steps", type=int, default=10000, help="number of optimization steps")
    parser.add_argument("--output_dir", type=str, default="output", help="path to store results")
    parser.add_argument("--pretrained_models_path", type=str, default="/homes/im311/repos/coalagan/wavegan/", help="path to store wavegan and coala pretrained models")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    id2tag = json.load(open('coala/id2token_top_1000.json', 'rb'))
    for id, tag in id2tag.items():
        play_my_words(tag, args)
