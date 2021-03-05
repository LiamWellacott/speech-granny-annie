from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import symbols, phonemes, make_symbols
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.io import load_checkpoint
from TTS.vocoder.utils.generic_utils import setup_generator

import torch

# runtime settings
use_cuda = False
use_gl = False

TTS_MODEL_PATH = './tts_models/tts_models-en-ljspeech-glow-tts-model.pth.tar'
TTS_CONFIG_PATH = './tts_models/tts_models-en-ljspeech-glow-tts-config.json'
VOCODER_MODEL_PATH = './tts_models/vocoder_models-en-ljspeech-mulitband-melgan-model.pth.tar'
VOCODER_CONFIG_PATH = './tts_models/vocoder_models-en-ljspeech-mulitband-melgan-config.json'
VOCODER_STATS_PATH = './tts_models/vocoder_models-en-ljspeech-mulitband-melgan-scale_stats.npy'

OUT_FILE = 'tts_out.wav'

# load configs
TTS_CONFIG = load_config(TTS_CONFIG_PATH)
VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)

def interpolate_vocoder_input(scale_factor, spec):
    """Interpolation to tolarate the sampling rate difference
    btw tts model and vocoder"""
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    spec = torch.nn.functional.interpolate(spec, scale_factor=scale_factor, mode='bilinear').squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def tts(model, text, CONFIG, use_cuda, ap, ap_vocoder, scale_factor, vocoder_model):
    # Run TTS
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap)
    

    # Run Vocoder
    if not use_gl:
        vocoder_input = ap_vocoder.normalize(mel_postnet_spec.T)
        if scale_factor[1] != 1:
            vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
        else:
            vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
    
    # format output
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()

    return alignment, mel_postnet_spec, stop_tokens, waveform

def load_model(sentence, TTS_MODEL_PATH, TTS_CONFIG, VOCODER_MODEL_PATH, VOCODER_CONFIG, use_cuda, OUT_FILE):
    ap = AudioProcessor(**TTS_CONFIG.audio)

    speakers = []
    speaker_id = None

    # if 'characters' in TTS_CONFIG.keys():
    #     symbols, phonemes = make_symbols(**c.characters)

    # load the model
    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), TTS_CONFIG)

    # load model state
    model, _ =  load_checkpoint(model, TTS_MODEL_PATH, use_cuda=use_cuda)
    model.eval()
    model.store_inverse()

    VOCODER_CONFIG.audio['stats_path'] = VOCODER_STATS_PATH
    
    # LOAD VOCODER MODEL
    vocoder_model = setup_generator(VOCODER_CONFIG)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL_PATH, map_location="cpu")["model"])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    # scale factor for sampling rate difference
    scale_factor = [1,  VOCODER_CONFIG['audio']['sample_rate'] / ap.sample_rate]
    print(f"scale_factor: {scale_factor}")

    ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])    
    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval()
    
    # faster speech
    model.length_scale = 0.8  # set speed of the speech. 
    model.noise_scale = 0.33  # set speech variationd

    align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, ap_vocoder, scale_factor, vocoder_model)

    ap.save_wav(wav, OUT_FILE)