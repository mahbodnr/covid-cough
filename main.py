import tensorflow as tf
import numpy as np
import librosa
from scipy.signal import stft as stft

from models import coughdetector, covidcough

SIZE=128
SILENCE = 0
SR = 44100
STRIDE_SIZE = 5000
SAMPLE_SIZE = 10000
#============
# SAMPLE_LENGTH = 0.25 #s
# STRIDE = .1  #s
# STRIDE_SIZE = int(np.ceil(SR*STRIDE))
# SAMPLE_SIZE = int(np.ceil(SR*SAMPLE_LENGTH))
#============
N_FFT = 2048
N_MELS = 60

def load_audio(path):
    signal, _ = librosa.load(path, sr=SR)
    return signal

def STFT(audio, STFT_info = (256, 200, 256)):
    _, _, Zxx = stft(audio, fs= 44100 ,window='hann', 
                            nperseg=STFT_info[0], noverlap=STFT_info[1],
                            nfft = STFT_info[2])
    db = librosa.amplitude_to_db(np.abs(Zxx), ref=np.max)
    return db[...,np.newaxis]

def melspectrogram(signal):
    signal = librosa.util.normalize(signal)
    spectro = librosa.feature.melspectrogram(
        signal,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT
    )
    spectro = librosa.power_to_db(spectro)
    spectro = spectro.astype(np.float32)
    return spectro[...,np.newaxis]

def process_audio(audio, transform, sample_size, stride_size):
    if type(audio)==str:  
        signal = load_audio(audio)
    else:
        signal = audio

    if len(signal) < sample_size:
        return []

    current = 0
    end = False
    features = []
    parts = []

    while not end:
        if current+sample_size > len(signal):
            sample = signal[len(signal)-sample_size:]
            end = True
        else:
            sample = signal[current:current+sample_size]
            current += stride_size

        features.append(transform(sample))
        parts.append(sample)

    return tf.convert_to_tensor(features), parts



def preprocess(audio_path):
    ### Find Cough Sound
    mel_features, parts = process_audio(audio_path, melspectrogram, 22000, 11000)
    coughs = np.squeeze(coughdetector(mel_features, False, None))
    cough_mask = coughs[:,0]> 0.5

    if not cough_mask.any():
        #No cough Found
        return None 
    ### Cough Sound to STFT
    STFT_features = []
    for part in parts:
        part_features, _ = process_audio(part, STFT, 10000, 2000)
        STFT_features.append(part_features)

    STFT_features = tf.convert_to_tensor(STFT_features)
    shape = tf.shape(STFT_features)
    return tf.reshape(STFT_features, [shape[0] * shape[1], *shape[2:]])    



def run(audio_path):
    #Read Audio
    extracted_features =  preprocess(audio_path)
    if type(extracted_features)!=type(None):
        #Send to Model
        prediction = np.squeeze(covidcough(extracted_features))
        pred_labels = np.argmax(prediction, axis= -1)
        if type(pred_labels)!=np.ndarray:
            pred_labels = np.array([pred_labels])
        if pred_labels.sum() > len(pred_labels)/2 :
            return 1, 'Suspicious to COVID-19'# + str(pred_labels)
        return 0, 'Healthy' # + str(prediction) 
    else:
        return 'No cough sound found in the audio.'
