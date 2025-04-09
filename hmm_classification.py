import os
import cv2
import numpy as np
from skimage.feature import hog
from hmmlearn import hmm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from sklearn.decomposition import PCA

def load_dataset(dataset_dir):
    """
    dataset_dir: Örneğin "train_images" veya "test_images".
    Her harf için ayrı bir alt klasördeki (klasör adı harf etiketini içerir)
    tüm görüntüleri ve etiketleri okuyarak döndürür.
    """
    images = []
    labels = []
    for label in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(label_dir, file_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(label)
    return images, labels

class CharacterHMMClassifier:
    def __init__(self, n_components=5, window_size=(28, 8), orientations=9, 
                 window_stride=4, covariance_type="full", n_iter=300,
                 tol=1e-1, pca_components=10):
        """
        n_components: HMM'in durum sayısı.
        window_size: (yükseklik, genişlik); 28 piksel yüksekliğindeki görüntüden özellik çıkarımı.
        orientations: HOG özniteliklerinin yön sayısı.
        window_stride: Pencere kaydırma adım uzunluğu. Overlap (örtüşmeli pencere) için pencere 
                       genişliğinden küçük seçilir.
        covariance_type: HMM içinde kullanılacak kovaryans tipi ("diag", "full", vb.).
        n_iter: HMM eğitiminde maksimum iterasyon sayısı.
        tol: EM algoritması için konverjans toleransı (log-likelihood değişim eşiği).
        pca_components: PCA ile indirgenecek öznitelik boyutu. Eğer None ise PCA kullanılmaz.
        """
        self.n_components = n_components
        self.window_size = window_size
        self.orientations = orientations
        self.window_stride = window_stride
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.pca_components = pca_components
        self.models = {}       # Her harf için HMM modeli saklanacak.
        self.pca_transforms = {}  # Her harf için PCA dönüşümü saklanacak.

    def preprocess_image(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return image

    def extract_features(self, image):
        """
        Görüntüyü ön işlemden geçirir, yeniden boyutlandırır ve overlapping pencere ile HOG öznitelikleri çıkarır.
        """
        image = self.preprocess_image(image)
        desired_height = self.window_size[0]
        if image.shape[0] != desired_height:
            scale = desired_height / image.shape[0]
            new_width = int(image.shape[1] * scale)
            image = cv2.resize(image, (new_width, desired_height))
        else:
            new_width = image.shape[1]
            
        features_seq = []
        win_width = self.window_size[1]
        for x in range(0, new_width - win_width + 1, self.window_stride):
            patch = image[:, x:x + win_width]
            hog_feature = hog(patch,
                              orientations=self.orientations,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(1, 1),
                              feature_vector=True)
            features_seq.append(hog_feature)
        return np.array(features_seq)

    def train(self, images, labels):
        """
        Her görüntüden çıkarılan öznitelikleri harf etiketlerine göre gruplar, 
        PCA ile boyut indirgeme uygular ve her harf için HMM modelini eğitir.
        """
        from collections import defaultdict
        sequences = defaultdict(list)
        for img, lbl in zip(images, labels):
            feat_seq = self.extract_features(img)
            sequences[lbl].append(feat_seq)
            
        # Her harf (label) için
        for lbl, seq_list in sequences.items():
            # seq_list: label'a ait öznitelik dizileri, her biri (seq_length, n_features)
            # X: tüm öznitelik dizilerinin yanyana koyulmuş matrisi
            X = np.vstack(seq_list)
            lengths = [len(seq) for seq in seq_list]
            
            # PCA dönüşümü uygulanacaksa
            if self.pca_components is not None:
                pca = PCA(n_components=self.pca_components, random_state=42)
                X_reduced = pca.fit_transform(X)
                self.pca_transforms[lbl] = pca
            else:
                X_reduced = X
            
            model = hmm.GaussianHMM(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    n_iter=self.n_iter,
                                    tol=self.tol,
                                    random_state=42)
            model.fit(X_reduced, lengths)
            self.models[lbl] = model
            print(f"Model for '{lbl}' trained on {len(seq_list)} sequences.")

    def predict(self, image):
        """
        Girdi görüntüden çıkarılan öznitelikler, her harf modelinin (ve varsa ilgili PCA dönüşümünün)
        yardımıyla değerlendirilir; en yüksek log-likelihood değerine sahip modelin etiketi döndürülür.
        """
        feat_seq = self.extract_features(image)
        best_score = -np.inf
        best_label = None
        # Her model için skor hesaplaması yapılırken, ilgili PCA dönüşümü uygulanıyor
        for lbl, model in self.models.items():
            feat_seq_transformed = feat_seq
            if self.pca_components is not None and lbl in self.pca_transforms:
                feat_seq_transformed = self.pca_transforms[lbl].transform(feat_seq)
            try:
                score = model.score(feat_seq_transformed)
            except Exception as e:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_label = lbl
        return best_label

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()

if __name__ == "__main__":
    train_dir = "train_images"
    test_dir = "test_images"

    train_images, train_labels = load_dataset(train_dir)
    print(f"Toplam {len(train_images)} eğitim görüntüsü yüklendi.")
    
    hmm_classifier = CharacterHMMClassifier(
        n_components=5,
        window_size=(28, 8),
        orientations=9,
        window_stride=4,
        covariance_type="full",
        n_iter=300,
        tol=1e-1,
        pca_components=10  # PCA ile boyut indirgeme: 10 bileşen
    )
    hmm_classifier.train(train_images, train_labels)
    
    test_images, test_labels = load_dataset(test_dir)
    print(f"Toplam {len(test_images)} test görüntüsü yüklendi.")
    
    pred_labels = []
    for img, true_lbl in zip(test_images, test_labels):
        pred_lbl = hmm_classifier.predict(img)
        pred_labels.append(pred_lbl)
        print(f"Gerçek: {true_lbl} -- Tahmin: {pred_lbl}")
    
    correct = sum(1 for p, t in zip(pred_labels, test_labels) if p == t)
    accuracy = 100 * correct / len(test_images)
    print(f"HMM tabanlı tahmin doğruluğu: {accuracy:.2f}%")
    
    labels_list = sorted(list(set(train_labels)))
    cm = confusion_matrix(test_labels, pred_labels, labels=labels_list)
    print("Classification Report:")
    print(classification_report(test_labels, pred_labels, labels=labels_list))
    
    plot_confusion_matrix(cm, classes=labels_list, title='Confusion Matrix - HMM Classifier')
    plt.savefig("confusion_matrix_hmm.jpeg", format="jpeg")
    plt.show()
