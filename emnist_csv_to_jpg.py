import os
import pandas as pd
import numpy as np
from PIL import Image

def emnist_csv_to_jpg_by_label(csv_file, output_folder, img_height=28, img_width=28, rotate=False):
    """
    EMNIST CSV dosyasını okuyarak, her etikete ait görüntüleri ayrı klasörlere kaydeder.
    
    csv_file: EMNIST CSV dosyasının yolu (ör. emnist-balanced-train.csv)
    output_folder: Görsellerin kaydedileceği ana klasör
    img_height, img_width: Görüntü boyutları (varsayılan: 28x28)
    rotate: EMNIST verilerinin yönünü düzeltmek için True yapılabilir.
    """
    
    # CSV dosyasını oku
    print(f"Okunacak CSV dosyası: {csv_file}")
    df = pd.read_csv(csv_file, header=None)
    print("CSV dosyası başarıyla okundu.")
    
    # EMNIST CSV formatı genellikle:
    # 1. Sütun = Etiket (label)
    # Sonraki 784 sütun = Piksel değerleri (28x28)
    
    # Verisetinin ilk sütunu etiket olsun:
    labels = df.iloc[:, 0].values
    # Piksel değerleri:
    pixel_data = df.iloc[:, 1:].values
    
    # Ana klasörü oluştur
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (label, row) in enumerate(zip(labels, pixel_data)):
        # Label için alt klasör oluştur
        label_folder = os.path.join(output_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        
        # Piksel verisini 28x28 boyutunda bir numpy dizisine dönüştür
        img_array = np.array(row, dtype=np.uint8).reshape(img_height, img_width)
        
        # Bazı EMNIST alt veri setleri ters (rotated) veya transpoze edilmiş olabilir.
        # rotate=True parametresi ile görselleri döndürebilirsiniz.
        if rotate:
            # 90 derece döndürmek için:
            img_array = np.rot90(img_array, k=1)
        
        # Gri tonlamalı (L) bir görüntü oluştur
        img = Image.fromarray(img_array, mode='L')
        
        # Dosya adını oluştur
        filename = os.path.join(label_folder, f"img_{i}.jpg")
        
        # Kaydet
        img.save(filename)
    
    print(f"Görüntüler başarıyla '{output_folder}' klasörüne etiket bazında kaydedildi.")

# Örnek kullanım:
csv_file_path = "emnist-balanced-test.csv"
output_folder = "emnist_test_output_images"
emnist_csv_to_jpg_by_label(csv_file_path, output_folder, rotate=False)
