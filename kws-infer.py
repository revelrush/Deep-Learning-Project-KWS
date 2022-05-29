import torch
import torchaudio
import numpy as np
from torchvision.transforms import ToTensor
from einops import rearrange
import sounddevice as sd
import time
import PySimpleGUI as sg
import librosa
from train import LitTransformer

CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
            'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
            'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


# make a dictionary from CLASSES to integers
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
print("Loading model checkpoint")
model = LitTransformer.load_from_checkpoint('KWS.ckpt')

idx_to_class = {i: c for i, c in enumerate(CLASSES)}
sample_rate = 16000
sd.default.device = 1
sd.default.samplerate = sample_rate
sd.default.channels = 1
sg.theme('DarkAmber')


transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                 n_fft=1024,
                                                 win_length=None,
                                                 hop_length=512,
                                                 n_mels=128,
                                                 power=2.0)

layout = [
    [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140),
             key='-OUTPUT-'), ],
    [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'), ],
    [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
]

window = sg.Window('KWS Inference', layout, location=(0, 0), resizable=True).Finalize()
window.Maximize()
window.BringToFront()

total_runtime = 0
n_loops = 0
while True:
    event, values = window.read(100)
    if event == sg.WIN_CLOSED:
        break

    waveform = sd.rec(sample_rate).squeeze()

    sd.wait()
    if waveform.max() > 1.0:
        continue
    start_time = time.time()
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
    mel = mel.unsqueeze(0)
    mel = rearrange(mel, 'b c h (p2 w) -> b (p2) (c h w)', p2=16)
    pred = model(mel)
    pred = torch.functional.F.softmax(pred, dim=1)
    max_prob = pred.max()
    elapsed_time = time.time() - start_time
    total_runtime += elapsed_time
    n_loops += 1
    ave_pred_time = total_runtime / n_loops
    if max_prob > 0.6:
        pred = torch.argmax(pred, dim=1)
        human_label = f"{idx_to_class[pred.item()]}"
        window['-OUTPUT-'].update(human_label)
        window['-OUTPUT-'].update(human_label)
        if human_label == "stop":
            window['-STATUS-'].update("Goodbye!")
            # refresh window
            window.refresh()
            time.sleep(1)
            break

    else:
        window['-OUTPUT-'].update("...")

    window['-TIME-'].update(f"{ave_pred_time:.2f} sec")

window.close()
