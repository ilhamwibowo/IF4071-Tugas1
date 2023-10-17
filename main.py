import numpy as np
from python_speech_features import mfcc, delta
import scipy.io.wavfile as wav
from multiprocessing import Pool
from pydub import AudioSegment
import time
import os

def extract_mfcc(audio_file, wav_file):
    track = AudioSegment.from_file(audio_file, format='m4a')
    file_handle = track.export(wav_file, format='wav')
    (rate, sig) = wav.read(wav_file)
    features = mfcc(sig, rate)
    return features

def extract_mfcc_wav(wav_file, n_fft=2048):
    (rate, signal) = wav.read(wav_file)
    mfcc_feat = mfcc(signal, rate, nfft=n_fft)
    delta_feat = delta(mfcc_feat, 2)
    delta_delta_feat = delta(delta_feat, 2)
    combined_features = np.concatenate((mfcc_feat, delta_feat, delta_delta_feat), axis=1)

    return combined_features

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

class AudioRecognizer:
    def __init__(self, template_data) -> None:
        self.template_date = template_data
        
    def dtw(self, input_tuple):

        f1, f2 = input_tuple[0], input_tuple[1]
        n, m = len(f1), len(f2)
        
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean_distance(f1[i - 1], f2[j - 1])

                cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])

        return cost_matrix[n, m]

    def predict(self, input_data):
        min = float("inf")
        prediction = None

        for word, template_mfcc in self.template_date.items():
            distance = self.dtw((input_data, template_mfcc))

            if distance < min:
                min = distance
                prediction = word

        return prediction

    def fast_predict(self, input_data):
        min = float("inf")
        prediction = None

        with Pool(processes=len(self.template_date)) as pool:
            # results = pool.starmap(dtw, input_it, template_it)
            results = pool.map(self.dtw, ((input_data, template_mfcc) for template_mfcc in self.template_date.values()))

        for i, (word, template_mfcc) in enumerate(self.template_date.items()):
            distance = results[i]

            if distance < min:
                min = distance
                prediction = word

        return prediction

if __name__ == '__main__':
    template_data = {
        'Bandung': extract_mfcc_wav('templates/bandung.wav'),
        'Semarang': extract_mfcc_wav('templates/semarang.wav'),
        'Palembang': extract_mfcc_wav('templates/palembang.wav'),
        'Medan': extract_mfcc_wav('templates/medan.wav'),
        'Banjarmasin': extract_mfcc_wav('templates/banjarmasin.wav'),
        'Palangkaraya': extract_mfcc_wav('templates/palangkaraya.wav'),
        'Manado': extract_mfcc_wav('templates/manado.wav'),
        'Kendari': extract_mfcc_wav('templates/kendari.wav'),
        'Fakfak': extract_mfcc_wav('templates/fakfak.wav'),
        'Ternate': extract_mfcc_wav('templates/ternate.wav'),
    }

    recognizer = AudioRecognizer(template_data)

    while True:
        print("Select an option:")
        print("1. Predict all test cases and show accuracy")
        print("2. Predict from a specific audio file")
        print("3. Exit")
        
        user_choice = input("Enter your choice (1/2/3): ")

        if user_choice == '1':
            test_folder = 'templates/Testcases'
            total_tests = 0
            correct_predictions = 0

            start_time = time.time()
            for folder_name in os.listdir(test_folder):
                folder_path = os.path.join(test_folder, folder_name)
                if os.path.isdir(folder_path):
                    for audio_file in os.listdir(folder_path):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(folder_path, audio_file)
                            input_data = extract_mfcc_wav(audio_path)
                            
                            recognized_word = recognizer.fast_predict(input_data)

                            total_tests += 1
                            if recognized_word == folder_name:
                                correct_predictions += 1

                            print(f"Tested audio: {audio_file}")
                            print("Predicted label:", recognized_word)
                            print("Actual label:", folder_name)
                            print('-' * 30)

            end_time = time.time()
            accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
            print("Parallel Execution Time:", end_time - start_time)
            print(f"Total Data Tested: {total_tests}")
            print(f"Total Accuracy: {accuracy:.2f}%")

        elif user_choice == '2':
            audio_file_path = input("Enter the path to the audio file (e.g., 'templates/audio.wav'): ")
            input_data = extract_mfcc_wav(audio_file_path)
            recognized_word = recognizer.fast_predict(input_data)
            print(f"Predicted label for '{audio_file_path}': {recognized_word}")

        elif user_choice == '3':
            break

        else:
            print("Invalid choice. Please enter '1', '2', or '3'.")







