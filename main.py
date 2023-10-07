import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from fastdtw import fastdtw
from multiprocessing import Pool
from pydub import AudioSegment
import time

def extract_mfcc(audio_file, wav_file):
    track = AudioSegment.from_file(audio_file, format='m4a')
    file_handle = track.export(wav_file, format='wav')
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

if __name__ == '__main__':
    template_data = {
        'bandung': extract_mfcc('templates/Template/Bandung.m4a', 'templates/bandung.wav'),
        'semarang': extract_mfcc('templates/Template/Semarang.m4a', 'templates/semarang.wav'),
        'palembang': extract_mfcc('templates/Template/Palembang.m4a', 'templates/palembang.wav'),
        'medan': extract_mfcc('templates/Template/Medan.m4a', 'templates/medan.wav'),
        'banjarmasin': extract_mfcc('templates/Template/Banjarmasin.m4a', 'templates/banjarmasin.wav'),
        'palangkaraya': extract_mfcc('templates/Template/Palangkaraya.m4a', 'templates/palangkaraya.wav'),
        'manado': extract_mfcc('templates/Template/Manado.m4a', 'templates/manado.wav'),
        'kendari': extract_mfcc('templates/Template/Kendari.m4a', 'templates/kendari.wav'),
        'fakfak': extract_mfcc('templates/Template/Fakfak.m4a', 'templates/fakfak.wav'),
        'ternate': extract_mfcc('templates/Template/Ternate.m4a', 'templates/ternate.wav'),
    }

    input_data = extract_mfcc('templates/Testcases/Ternate/Recording.m4a', 'templates/Testcases/test1.wav')
    
    start_time = time.time()
    recognized_word = fast_predict(input_data, template_data)
    end_time = time.time() 
    print("Parallel Execution Time :", end_time - start_time)
    print("Kata yang diucapkan:", recognized_word)

    start_time = time.time()
    recognized_word = predict(input_data, template_data)
    end_time = time.time() 
    print("Serial Execution Time:", end_time - start_time)
    print("Kata yang diucapkan:", recognized_word)






