import os
import string

# İngilizce alfabenin büyük harflerini alıyoruz
alphabet = list(string.ascii_uppercase)

# 10'dan başlayarak, her sayıyı alfabetik harfle eşleştiriyoruz:
# Örneğin, "10" -> "A", "11" -> "B", ..., "35" -> "Z"
mapping = {str(i): letter for i, letter in enumerate(alphabet)}

# İşlenecek ana klasörlerin listesi
directories = ["a-z_train_output_images", "a-z_test_output_images"]

for main_dir in directories:
    # Belirtilen ana klasördeki tüm öğeleri kontrol ediyoruz
    for folder_name in os.listdir(main_dir):
        old_folder_path = os.path.join(main_dir, folder_name)
        # Sadece klasörleri işleyelim
        if os.path.isdir(old_folder_path):
            # Eğer klasör ismi mapping içinde varsa, yeni ismi belirle
            if folder_name in mapping:
                new_folder_name = mapping[folder_name]
                new_folder_path = os.path.join(main_dir, new_folder_name)
                os.rename(old_folder_path, new_folder_path)
                print(f"{old_folder_path} -> {new_folder_path}")
            else:
                print(f"{folder_name} mapping içinde bulunamadı.")
