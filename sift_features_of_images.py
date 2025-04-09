import os
import cv2
import numpy as np
import pandas as pd

def process_image_dir_sift(root_dir, sift, bow_extractor):
    """
    Belirtilen kök klasör altındaki alt klasörlerdeki her resmi okuyup,
    BOW yöntemiyle SIFT tabanlı histogram (öznitelik vektörü) çıkarır.
    Alt klasör adı, resmin etiket değeri olarak kabul edilir.
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
                # SIFT ile tespit edilen keypoint'leri al
                keypoints = sift.detect(image, None)
                # Eğer keypoint bulunamazsa, histogramı sıfırlardan oluştur
                hist = bow_extractor.compute(image, keypoints)
                if hist is None:
                    # SIFT keypoint bulunamazsa, 0'lardan histogram oluştur
                    hist = np.zeros((1, bow_extractor.descriptorSize()), dtype=np.float32)
                data.append(hist.flatten())
                labels.append(label)
    return np.array(data), np.array(labels)

if __name__ == '__main__':
    # Ayarlar: Görsel kelime sayısı (vocabulary boyutu)
    k = 512  # Örneğin, 50 görsel kelime; bu değer öznitelik boyutunu belirler.

    # SIFT tanımlayıcıyı oluştur
    sift = cv2.SIFT_create()
    
    # Eğitim verisindeki SIFT tanımlayıcılarından tüm descriptorleri toplayalım
    train_dir = "train_images"
    descriptor_list = []
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                kp, des = sift.detectAndCompute(image, None)
                if des is not None:
                    descriptor_list.append(des)
    if len(descriptor_list) == 0:
        raise ValueError("Hiçbir descriptor bulunamadı. Eğitim verisini kontrol edin.")
    all_descriptors = np.vstack(descriptor_list)
    
    # Vocabulary oluşturmak için k-means clustering (BOW Trainer) kullanıyoruz
    bow_trainer = cv2.BOWKMeansTrainer(k)
    vocabulary = bow_trainer.cluster(all_descriptors)
    
    # BOW (Bag-of-Visual-Words) öznitelik çıkarıcıyı oluşturuyoruz
    bf = cv2.BFMatcher(cv2.NORM_L2)
    bow_extractor = cv2.BOWImgDescriptorExtractor(sift, bf)
    bow_extractor.setVocabulary(vocabulary)
    
    # Eğitim verileri için SIFT-BOW özniteliklerini çıkarıp CSV'ye kaydedelim
    train_features, train_labels = process_image_dir_sift(train_dir, sift, bow_extractor)
    df_train = pd.DataFrame(train_features)
    df_train['label'] = train_labels
    df_train.to_csv("sift_512_train_features.csv", index=False)
    print("Eğitim verileri için öznitelikler ilgili dosyaya kaydedildi.")
    
    # Test verileri için aynı vocabulary kullanılarak öznitelik çıkarımı
    test_dir = "test_images"
    test_features, test_labels = process_image_dir_sift(test_dir, sift, bow_extractor)
    df_test = pd.DataFrame(test_features)
    df_test['label'] = test_labels
    df_test.to_csv("sift_512_test_features.csv", index=False)
    print("Test verileri için öznitelikler ilgili dosyaya kaydedildi.")