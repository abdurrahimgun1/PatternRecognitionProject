import pandas as pd
import numpy as np
from PIL import Image
import os

csv_file = "A_Z_Handwritten_Data_test.csv"  # CSV dosyasının yolu

# CSV dosyasını header olmadan oku (etiket sütunu dosyanın ilk sütunu olarak gelecek)
df = pd.read_csv(csv_file, header=None)

output_folder = "a-z_test_output_images"
os.makedirs(output_folder, exist_ok=True)

img_height = 28
img_width = 28

# Her satırdaki ilk değer label, kalan değerler piksel verileri
for idx, row in df.iterrows():
    # İlk sütundaki etiket
    label = row.iloc[0]
    # Geri kalan sütunlar piksel verileri
    pixels = row.iloc[1:].values

    # Piksel verilerini 28x28 boyutunda diziye dönüştür
    img_array = np.array(pixels, dtype=np.uint8).reshape(img_height, img_width)
    
    # Görüntüyü oluştur
    img = Image.fromarray(img_array, mode='L')
    
    # Etikete göre klasör oluştur
    label_folder = os.path.join(output_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)
    
    # Görüntüye benzersiz bir isim ver (örneğin, img_0.jpg)
    img_path = os.path.join(label_folder, f'image_{idx}.jpg')
    img.save(img_path)

print("Görüntüler başarıyla etiket bazında klasörlere kaydedildi.")
