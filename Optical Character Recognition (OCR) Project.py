import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import cv2 # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
from scipy import ndimage

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD


folder = sys.argv[1]
total_hemming_distance = 0

# Učitavanje CSV datoteke
csv_file_path = os.path.join(folder, 'res.csv')
csv_data = pd.read_csv(csv_file_path)

# Definisanje Hemingove distance
def hemming_distance(str1, str2):
    len_diff = abs(len(str1) - len(str2))
    if len(str1) > len(str2):
        str1 = str1[:len(str2)]
    elif len(str2) > len(str1):
        str2 = str2[:len(str1)]
    dist = sum(1 for x, y in zip(str1, str2) if x != y)
    return dist + len_diff

# Mapiranje naziva datoteka sa njihovim tekstovima
expected_texts = dict(zip(csv_data['file'], csv_data['text']))

#koraci 1-3

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        
def dilate(image):
    kernel = np.ones((12, 3)) 
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) 
    return cv2.erode(image, kernel, iterations=1)

#korak 4
def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

#korak 5
def scale_to_range(image):
    return image/255

#flattitanje
def matrix_to_vector(image):
    return image.flatten()

#priprema za trening
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

#koraci 6 i 7

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

#korak 8-odredjivanje pobednickog neurona
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

#prikaz rezultata sa razmacima!!
def display_result_with_spaces_and_text(outputs, alphabet, k_means, max_space_size, true_text):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        space_size = k_means.cluster_centers_[k_means.labels_[idx]].squeeze()
        if space_size > max_space_size:
            result += ' '
        result += alphabet[winner(output)]
    # Izračunavanje Hemingove distance
    hemming_dist = hemming_distance(result, true_text)
    return result, hemming_dist
     
#odredjivanje regiona od interesa
def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(y<260 and w>5):
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) 
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def regions_similarity(img1, img2):
    # Normalizacija i thresholding za poboljšanje poređenja
    _, img1_threshold = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, img2_threshold = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Uklanjanje šuma
    img1_blur = cv2.GaussianBlur(img1_threshold, (3, 3), 0)
    img2_blur = cv2.GaussianBlur(img2_threshold, (3, 3), 0)
    
    # Poređenje slika
    similarity = np.sum(img1_blur == img2_blur) / img1_blur.size
    
    return similarity

def are_regions_similar(img1, img2, threshold=0.95):
    return regions_similarity(img1, img2) > threshold

unique_regions = []
image_folder =  os.path.join(folder, 'pictures')

# Prolazak kroz sve slike u direktorijumu sa slikama
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_color = load_image(image_path)
        img = image_bin(image_gray(image_color))
        img = invert(img)
        img = erode(dilate(img))
        image_orig, regions, region_distances = select_roi_with_distances(image_color.copy(), img)
        # display_image(image_orig)
        # plt.show()
        for region in regions:
            # Provera da li je region sličan nekom već postojećem
            if not any(are_regions_similar(region, unique_region) for unique_region in unique_regions):
                unique_regions.append(region)

# print(f"Broj jedinstvenih regiona: {len(unique_regions)}")


alphabet = ['k', 'l', 'e', 'p', 'n', 'ú', 'č', 'a', 's', 't', 'í', 'c', 'o', 'ť', 'j', 'š', 'v', 'z', 'm', 'ě', 'd', 'h', 'á', 'ý', 'i', 'r', 'é', 'ž', 'y', 'b']
inputs = prepare_for_ann(unique_regions)
outputs = convert_output(alphabet)
ann = create_ann(output_size=30)
ann = train_ann(ann, inputs, outputs, epochs=2000)

# Prolazak kroz sve slike u direktorijumu sa slikama i izračunavanje Hemingovog rastojanja
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_color = load_image(image_path)
        img = image_bin(image_gray(image_color))
        img = invert(img)
        img = erode(dilate(img))
        selected_regions, letters, distances = select_roi_with_distances(image_color.copy(), img)

        # Izračunavanje K-Means
        distances = np.array(distances).reshape(len(distances), 1)
        k_means = KMeans(n_clusters=2, n_init=10)
        k_means.fit(distances)

        # Izračunavanje rezultata i Hemingove distance
        inputs = prepare_for_ann(letters)
        results = ann.predict(np.array(inputs, np.float32))
        predicted_text, hemming_dist = display_result_with_spaces_and_text(results, alphabet, k_means, 13, expected_texts[filename])
        total_hemming_distance += hemming_dist

        print(f"File: {filename}, Predicted: {predicted_text}, Expected: {expected_texts[filename]}, Hemming Distance: {hemming_dist}")
print(f"Ukupna Hemingova distanca: {total_hemming_distance}")


