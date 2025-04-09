import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

def extract_hog_features(image, pixels_per_cell=(7, 7), cells_per_block=(2, 2), orientations=9):
    """
    Verilen gri tonlamalı görüntü için HOG özniteliklerini çıkarır.
    Parametreleri değiştirerek öznitelik boyutunu azaltıyoruz.
    Bu ayarlarla 28x28 boyutundaki bir resim için öznitelik vektör boyutu 324 olur.
    """
    features = hog(image,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    return features

def process_image_dir(root_dir):
    """
    Kök klasör altındaki tüm alt klasörlerdeki resimleri okuyup, HOG özniteliklerini çıkarır.
    Alt klasör ismi, resmin etiket değeri olarak kabul edilir.
    """
    data = []
    labels = []
    for label in os.listdir(root_dir):
        label_dir = os.path.join(root_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Resim okunamadı: {img_path}")
                    continue
                features = extract_hog_features(image)
                data.append(features)
                labels.append(label)
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    # İşlenecek ana klasörler ve çıkış dosyaları
    datasets = {
        "train": "train_images",
        "test": "test_images"
    }
    
    for key, folder in datasets.items():
        features, labels = process_image_dir(folder)
        df_features = pd.DataFrame(features)
        # Etiket sütunu ekleniyor: Alt klasör adı, yani harf
        df_features['label'] = labels
        
        output_csv = f"hog_{key}_features.csv"
        df_features.to_csv(output_csv, index=False)
        print(f"{key.capitalize()} verisi için öznitelikler '{output_csv}' dosyasına kaydedildi.")
