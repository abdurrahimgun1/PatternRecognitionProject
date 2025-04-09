import pandas as pd
import numpy as np

# CSV dosyasını header olmadan oku; ilk sütun etiket (grupby değeri), geri kalan sütunlar piksel verileri
csv_file = "A_Z Handwritten Data.csv"
df = pd.read_csv(csv_file, header=None)

# İlk sütuna göre gruplandır (grupby değeri: etiket)
groups = df.groupby(0)

train_list = []
test_list = []

# Her grup için verileri karıştırıp %85 train, %15 test olarak ayırıyoruz
for label, group in groups:
    group = group.sample(frac=1, random_state=42)  # Grupları rastgele karıştır
    split_index = int(len(group) * 0.85)
    train_group = group.iloc[:split_index]
    test_group = group.iloc[split_index:]
    
    train_list.append(train_group)
    test_list.append(test_group)

# Her gruptan ayrılan verileri birleştirerek genel train ve test dataframe'lerini oluşturuyoruz
df_train = pd.concat(train_list, ignore_index=True)
df_test = pd.concat(test_list, ignore_index=True)

# Dataframe'leri CSV olarak kaydetme
# Burada ilk sütun (etiket) da veri olarak kaydedilir.
df_train.to_csv("A_Z_Handwritten_Data_train.csv", index=False, header=False)
df_test.to_csv("A_Z_Handwritten_Data_test.csv", index=False, header=False)

print("Veriler başarıyla iki ayrı CSV dosyasına ayrıldı: %85 train, %15 test. Etiket sütunu da kaydedildi.")
