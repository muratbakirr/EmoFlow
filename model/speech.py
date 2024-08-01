# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import librosa.display
from IPython.display import Audio
import os
import librosa
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

base_path = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the script's directory
data_path = os.path.join(base_path, '../data/last_features.csv')
model_path = os.path.join(base_path, '../model/last_version_model.h5')


def main(args):
  # load feaures data
  feature = get_features(args.path)
  t, encoder = process_features(feature, args.feauture_path)
  y_pred = predict(t, args.model_path, encoder)
  print(str(y_pred[0][0]))

def extract_features(data,sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    feature = np.array(res1)

    return feature

def process_features(feature,feauture_path):
  Features = pd.read_csv(feauture_path)
    
  X = Features.iloc[: ,:-1].values
  Y = Features['labels'].values

  encoder = OneHotEncoder()
  Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()  
  x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)

  feature = pd.DataFrame(feature).T
  t = feature.values
  t = scaler.transform(t)
  t = np.expand_dims(t, axis=2)
  
  return t, encoder

  
def predict(t, model_path, encoder):
  model = load_model(model_path)
  pred_test = model.predict(t)
  y_pred = encoder.inverse_transform(pred_test)
  
  return y_pred

def parse_arg():
  parser = argparse.ArgumentParser()

  parser.add_argument('--feauture_path', type=str, default=data_path)
  parser.add_argument('--model_path', type=str, default=model_path)
  parser.add_argument('--path', type=str, dest='path')

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_arg()
  main(args)
