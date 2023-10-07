import numpy as np
from python_speech_features import mfcc
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

def extract_mfcc_wav(wav_file):
    (rate, sig) = wav.read(wav_file)
    features = mfcc(sig, rate)
    return features

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

# def dtw(f1, f2):
#     n, m = len(f1), len(f2)

#     cost_matrix = np.full((n + 1, m + 1), np.inf)
#     cost_matrix[0, 0] = 0

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             # cost = manhattan_distance(s1[i - 1], s2[j - 1])
#             cost = euclidean_distance(f1[i - 1], f2[j - 1])

#             cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])

#     return cost_matrix[n, m]

# pool.map cuman bisa 1 argumen
def dtw(input_tuple):

    f1, f2 = input_tuple[0], input_tuple[1]
    n, m = len(f1), len(f2)
    
    # print(f1)
    # print(f2)

    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # cost = manhattan_distance(s1[i - 1], s2[j - 1])
            cost = euclidean_distance(f1[i - 1], f2[j - 1])

            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])

    return cost_matrix[n, m]

def predict(input_data, template_data):
    min = float("inf")
    prediction = None

    for word, template_mfcc in template_data.items():
        distance = dtw((input_data, template_mfcc))

        if distance < min:
            min = distance
            prediction = word

    return prediction

def fast_predict(input_data, template_data):
    min = float("inf")
    prediction = None

    with Pool(processes=len(template_data)) as pool:
        # results = pool.starmap(dtw, input_it, template_it)
        results = pool.map(dtw, ((input_data, template_mfcc) for template_mfcc in template_data.values()))

    for i, (word, template_mfcc) in enumerate(template_data.items()):
        distance = results[i]

        if distance < min:
            min = distance
            prediction = word

    return prediction

# if __name__ == '__main__':
#     template_data = {
#         'Bandung': extract_mfcc('templates/Template/Bandung.m4a', 'templates/bandung.wav'),
#         'Semarang': extract_mfcc('templates/Template/Semarang.m4a', 'templates/semarang.wav'),
#         'Palembang': extract_mfcc('templates/Template/Palembang.m4a', 'templates/palembang.wav'),
#         'Medan': extract_mfcc('templates/Template/Medan.m4a', 'templates/medan.wav'),
#         'Banjarmasin': extract_mfcc('templates/Template/Banjarmasin.m4a', 'templates/banjarmasin.wav'),
#         'Palangkaraya': extract_mfcc('templates/Template/Palangkaraya.m4a', 'templates/palangkaraya.wav'),
#         'Manado': extract_mfcc('templates/Template/Manado.m4a', 'templates/manado.wav'),
#         'Kendari': extract_mfcc('templates/Template/Kendari.m4a', 'templates/kendari.wav'),
#         'Fakfak': extract_mfcc('templates/Template/Fakfak.m4a', 'templates/fakfak.wav'),
#         'Ternate': extract_mfcc('templates/Template/Ternate.m4a', 'templates/ternate.wav'),
#     }

#     test_folder = 'templates/Testcases'
#     total_tests = 0
#     correct_predictions = 0

#     for folder_name in os.listdir(test_folder):
#         folder_path = os.path.join(test_folder, folder_name)
#         if os.path.isdir(folder_path):
#             for audio_file in os.listdir(folder_path):
#                 if audio_file.endswith('.m4a'):
#                     audio_path = os.path.join(folder_path, audio_file)
#                     input_data = extract_mfcc(audio_path, f'templates/Testcases/{folder_name}/{audio_file[:-4]}.wav')
                    
#                     start_time = time.time()
#                     recognized_word = fast_predict(input_data, template_data)
#                     end_time = time.time()

#                     total_tests += 1
#                     if recognized_word == folder_name:
#                         correct_predictions += 1

#                     print(f"Tested audio: {audio_file}")
#                     print("Predicted label:", recognized_word)
#                     print("Actual label:", folder_name)
#                     print("Parallel Execution Time:", end_time - start_time)
#                     print('-' * 30)

#     accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
#     print(f"Total Accuracy: {accuracy:.2f}%")

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
                    
                    recognized_word = fast_predict(input_data, template_data)

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






