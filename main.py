import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from fastdtw import fastdtw
from multiprocessing import Pool
import time

def extract_mfcc(audio_file):
    (rate, sig) = wav.read(audio_file)
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
        'halo': extract_mfcc('templates/halo.wav'),
        'saya': extract_mfcc('templates/saya.wav'),
        'a': extract_mfcc('templates/halo.wav'),
        'b': extract_mfcc('templates/saya.wav'),
        'c': extract_mfcc('templates/halo.wav'),
        'd': extract_mfcc('templates/saya.wav'),
        'e': extract_mfcc('templates/halo.wav'),
        'f': extract_mfcc('templates/saya.wav'),
        'g': extract_mfcc('templates/halo.wav'),
        'h': extract_mfcc('templates/saya.wav'),
    }

    input_data = extract_mfcc('templates/saya.wav')
    
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






