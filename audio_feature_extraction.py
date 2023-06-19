import librosa
import os
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")


def extract_features(directory, json_path, frame_size=2048, hop_length=126, num_frames=20):

    features = {
        "genres_labels" : [],
        "mel_spectrograms": [],
        "targets": []
    }

    # Iterate over genre directories
    for genre_code, genre_dir in enumerate(os.scandir(directory)):
        if genre_dir.is_dir():
            genre_label = genre_dir.name
            features["genres_labels"].append(genre_label)

            print(f"Processing files in the {genre_label} genre")

            for audio_file in os.scandir(genre_dir.path): # going through each folder
                if audio_file.is_file(): # checking if the file is an audio file
                    audio_data, sample_rate = librosa.load(audio_file.path) # reding the audio file

                     # creating frames for each audio file
                    for i in range(0, min(len(audio_data) - frame_size + 1, num_frames * hop_length), hop_length):
                        frame = audio_data[i : i + frame_size]  # Extract current frame

                        # Compute Mel spectrogram for current  frame
                        mel_spectrogram = librosa.feature.melspectrogram(y=frame, sr=sample_rate)
                        features["mel_spectrograms"].append(mel_spectrogram.tolist())
                        y=np.zeros((10), dtype=np.float32)
                        y[genre_code] = 1.0
                        features["targets"].append(y.tolist())
                        
    feat=np.array(features["mel_spectrograms"])
    tar = np.array(features["targets"])
    print(feat.shape)
    print(tar.shape)

    # Save features to JSON
    with open(json_path, "w") as fp:
        json.dump(features, fp, indent=4)

    print("Extraction Completed!")
