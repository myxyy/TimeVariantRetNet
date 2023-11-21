import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import hydra
from hydra.utils import instantiate

np.set_printoptions(threshold=np.inf)

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):
    devices = cfg.predict.devices
    model = instantiate(cfg.model)
    model = model(devices=devices)
    model.load_state_dict(torch.load(cfg.predict.weight)['state_dict'])
    model.eval()
    context_len = cfg.predict.context_len
    out_length = cfg.predict.max_len
    vocab_size = 256
    dtype = model.dtype
    temperature = cfg.predict.temperature

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#parameter:{num_parameters}")

    def predict(prompt):
        prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(int)).clone().to(devices[0])
        prompt_len = len(prompt)
        prompt = torch.nn.functional.pad(prompt, (0, out_length-prompt_len), 'constant', 0)

        beam_width = 1
        model.reset_hidden()

        current_len = 0
        start = 0
        model.set_is_refresh(True)
        prompt_beam = prompt.repeat(beam_width, 1)
        while current_len < prompt_len:
            x = prompt_beam[:,current_len:current_len+context_len]
            x = nn.functional.one_hot(x.long(), vocab_size).to(dtype)
            if (prompt_len - current_len <= context_len):
                model.set_is_refresh(False)
                predict_init = model(x) # (1, context_len, vocab_size)
                #predict_init_i = predict_init.view(context_len, vocab_size)[prompt_len - current_len -1].topk(beam_width)
                predict_init_i = torch.multinomial(nn.Softmax(dim=1)(predict_init[:,prompt_len-current_len-1,:]/temperature), 1)
                prompt_beam[:,prompt_len] = predict_init_i
                current_len = prompt_len
            else:
                model.set_is_refresh(True)
                model(x)
                current_len += context_len
                start += context_len

        while current_len < out_length:
            model.set_is_refresh(current_len % context_len == 0)
            x = prompt_beam[:,start:start+context_len]
            x = nn.functional.one_hot(x.long(), vocab_size).to(dtype)
            predict_beam = model(x).to(devices[0])
            #_, predict_beam_i = predict_beam[:,current_len-1-start,:].reshape(beam_width * vocab_size).topk(beam_width)
            predict_beam_i = torch.multinomial(nn.Softmax(dim=1)(predict_beam[:,current_len-1-start,:]/temperature), 1)
            #prompt_beam = prompt_beam[torch.div(predict_beam_i, vocab_size, rounding_mode='floor')]
            #prompt_beam[:,current_len] = predict_beam_i % vocab_size 
            prompt_beam[:,current_len] = predict_beam_i

            predict = prompt_beam[0]
            predict = predict.cpu().numpy().astype(dtype='uint8')
            predict = predict.tobytes().decode('utf-8', 'replace')

            current_len += 1

            print(f'{current_len} Bytes')
            print(predict)

            if current_len % context_len == 1 or context_len == 1:
                start = start + context_len

        predict = prompt_beam[0]
        predict = predict.cpu().numpy().astype(dtype='uint8')
        predict = predict.tobytes().decode('utf-8', 'replace')
        return predict

    while True:
        prompt = input('prompt:')
        predict(prompt)

if __name__ == '__main__':
    main()