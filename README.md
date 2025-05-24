# Laporan Proyek Machine Learning - [Nama Anda]

## Domain Proyek

Kualitas udara merupakan salah satu faktor utama yang mempengaruhi kesehatan masyarakat dan kelestarian lingkungan. Polusi udara yang tinggi dapat menyebabkan berbagai masalah kesehatan seperti penyakit pernapasan, gangguan kardiovaskular, hingga kematian dini. Oleh karena itu, pengawasan dan pengelolaan kualitas udara menjadi sangat penting untuk mencegah dampak negatif tersebut.

Permasalahan yang dihadapi adalah bagaimana mengklasifikasikan kualitas udara secara akurat berdasarkan berbagai parameter lingkungan dan demografis. Dengan perkembangan teknologi machine learning, klasifikasi kualitas udara dapat dilakukan secara otomatis dan efektif menggunakan data kuantitatif seperti konsentrasi partikel PM2.5, PM10, kadar gas NO2, SO2, CO, serta faktor lain seperti suhu, kelembaban, dan kepadatan penduduk.

Beberapa studi telah menunjukkan bahwa metode machine learning mampu meningkatkan akurasi prediksi kualitas udara dibandingkan metode tradisional [1][2]. Pendekatan ini sangat relevan mengingat ketersediaan data yang semakin melimpah dan kebutuhan sistem pengawasan yang cepat serta andal.

Dataset yang digunakan dalam proyek ini berisi 5000 sampel yang mencakup berbagai variabel penting lingkungan dan demografi, serta kelas target kualitas udara yang dikategorikan menjadi *Good*, *Moderate*, *Poor*, dan *Hazardous*. Dataset ini dapat diakses secara publik melalui [Kaggle - Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/data).

---

### Referensi

[1] Y. Zheng, F. Liu, and H. Hsieh, “Predicting Air Quality with Random Forest Regression,” *Environmental Modelling & Software*, vol. 142, 2021.  
[2] Q. Liu, X. Wang, and L. Zhang, “Machine Learning Models for Air Pollution Forecasting: A Comparative Study,” *Atmospheric Environment*, vol. 270, 2022.

## Business Understanding

### Problem Statements
- Bagaimana cara mengklasifikasikan kualitas udara ke dalam kategori *Good*, *Moderate*, *Poor*, dan *Hazardous* berdasarkan data kuantitatif lingkungan dan demografis?  
- Bagaimana membangun model machine learning yang akurat dan andal untuk mendukung pengawasan kualitas udara di berbagai wilayah?  
- Bagaimana mengidentifikasi faktor-faktor lingkungan utama yang mempengaruhi kualitas udara agar pengambilan keputusan dapat lebih tepat sasaran?

### Goals
- Mengembangkan model klasifikasi kualitas udara yang dapat mengelompokkan data menjadi kelas yang tepat dengan akurasi tinggi.  
- Mampu mendeteksi kondisi udara berbahaya (*Hazardous*) dengan tingkat presisi dan recall yang optimal untuk meminimalkan risiko kesehatan masyarakat.  
- Melakukan analisis fitur untuk mengetahui variabel mana yang paling berpengaruh terhadap kualitas udara.

### Solution Statements
- Menggunakan tiga algoritma machine learning: **Random Forest**, **Support Vector Machine (SVM)**, dan **K-Nearest Neighbors (KNN)** untuk membandingkan performa dalam klasifikasi kualitas udara.  
- Melakukan evaluasi model berdasarkan metrik klasifikasi yang relevan seperti akurasi, precision, recall, dan F1-score untuk memilih model terbaik.  
- Analisis fitur penting dilakukan khususnya pada model Random Forest untuk memahami kontribusi variabel terhadap prediksi kualitas udara.


## Data Understanding

Dataset yang digunakan dalam proyek ini berfokus pada penilaian kualitas udara di berbagai wilayah. Dataset ini terdiri dari 5000 sampel data yang lengkap dan tidak memiliki nilai yang hilang (missing values). Data ini mencakup berbagai faktor lingkungan serta demografis yang berkontribusi terhadap tingkat polusi udara.

Dataset dapat diunduh secara publik melalui tautan berikut:  
[Kaggle - Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/data)

### Variabel-variabel pada dataset Air Quality adalah sebagai berikut:
- **Temperature (°C)**: Rata-rata suhu di wilayah tersebut.  
- **Humidity (%)**: Kelembaban relatif di wilayah pengukuran.  
- **PM2.5 (µg/m³)**: Konsentrasi partikel halus berukuran ≤ 2.5 mikrometer.  
- **PM10 (µg/m³)**: Konsentrasi partikel kasar berukuran ≤ 10 mikrometer.  
- **NO2 (ppb)**: Konsentrasi nitrogen dioksida.  
- **SO2 (ppb)**: Konsentrasi sulfur dioksida.  
- **CO (ppm)**: Konsentrasi karbon monoksida.  
- **Proximity to Industrial Areas (km)**: Jarak lokasi pengukuran ke kawasan industri terdekat.  
- **Population Density (people/km²)**: Kepadatan penduduk di wilayah tersebut.  
- **Air Quality (Kategori)**: Variabel target, dengan kategori:  
  - *Good*: Udara bersih dengan tingkat polusi rendah.  
  - *Moderate*: Kualitas udara dapat diterima meskipun terdapat polutan.  
  - *Poor*: Kualitas udara buruk, berpotensi mengganggu kelompok sensitif.  
  - *Hazardous*: Udara sangat tercemar, risiko serius terhadap kesehatan.

### Exploratory Data Analysis (EDA) dan Visualisasi Data

Beberapa langkah eksplorasi data dilakukan untuk memahami karakteristik dataset:  

- **Statistik Deskriptif** menunjukkan rentang dan distribusi tiap fitur, misalnya suhu berkisar antara 13.4°C hingga 58.6°C, kelembaban 36% hingga 128.1%, serta nilai PM2.5 dan PM10 yang memiliki nilai maksimum cukup tinggi (295 dan 315.8 µg/m³).  
- **Distribusi kelas target** diperiksa menggunakan visualisasi bar plot untuk memastikan keseimbangan data antar kelas.  
- **Visualisasi korelasi antar fitur** dilakukan menggunakan heatmap, yang menunjukkan korelasi tinggi antara PM2.5 dan PM10 (0.97), serta hubungan negatif antara jarak ke kawasan industri dengan tingkat polutan.

Visualisasi dan analisis awal ini sangat membantu dalam memahami hubungan antar fitur dan peran masing-masing dalam menentukan kualitas udara, sekaligus mendukung proses feature selection dan pemodelan selanjutnya.

---

## Data Preparation

Pada tahap ini, dilakukan beberapa proses persiapan data untuk memastikan dataset siap digunakan dalam pemodelan machine learning. Berikut langkah-langkah yang diterapkan secara berurutan:

1. **Penanganan Missing Value**  
   Dataset dicek untuk nilai yang hilang (missing values) dengan metode `.isnull().sum()`. Pada dataset ini tidak ditemukan missing value, sehingga tidak diperlukan imputasi atau penghapusan data.

2. **Encoding Variabel Target**  
   Variabel target *Air Quality* yang berupa kategori teks diubah menjadi format numerik menggunakan `LabelEncoder`. Proses ini penting agar algoritma machine learning dapat memproses data target dalam bentuk numerik yang sesuai.

3. **Pemisahan Fitur dan Target**  
   Dataset dibagi menjadi variabel fitur (independent variables) dan target (dependent variable). Variabel fitur terdiri dari semua kolom numerik yang merepresentasikan kondisi lingkungan dan demografis, sementara target adalah kelas kualitas udara.

4. **Normalisasi Fitur**  
   Seluruh fitur numerik dinormalisasi menggunakan `StandardScaler` untuk mengubah skala data agar memiliki rata-rata 0 dan standar deviasi 1. Normalisasi ini membantu algoritma machine learning, khususnya Random Forest, untuk bekerja lebih efisien dan mencegah fitur dengan rentang nilai besar mendominasi proses pelatihan.

5. **Pembagian Data Train-Test**  
   Data dibagi menjadi data latih dan data uji dengan perbandingan 70% untuk pelatihan dan 30% untuk pengujian menggunakan fungsi `train_test_split` dengan stratifikasi pada target. Stratifikasi memastikan distribusi kelas pada kedua set tetap seimbang sehingga model dapat belajar dengan representasi yang tepat.

---

### Alasan Penting Tahapan Data Preparation

- **Kebersihan Data:** Menghindari masalah seperti missing value yang dapat menyebabkan error atau bias pada model.  
- **Format Data Sesuai:** Encoding target memungkinkan algoritma yang hanya menerima input numerik untuk berjalan tanpa hambatan.  
- **Skala Data Seragam:** Normalisasi menjaga agar fitur tidak berat sebelah saat pelatihan model, meskipun Random Forest relatif toleran terhadap skala, normalisasi tetap baik untuk kestabilan.  
- **Distribusi Kelas Seimbang:** Pembagian data dengan stratifikasi penting agar evaluasi model pada data uji mencerminkan performa sebenarnya di setiap kelas.

Proses ini memastikan data dalam kondisi optimal untuk menghasilkan model machine learning yang akurat dan dapat diandalkan.

## Modeling

Dalam proyek ini, digunakan tiga algoritma machine learning untuk menyelesaikan masalah klasifikasi kualitas udara, yaitu **Random Forest**, **Support Vector Machine (SVM)**, dan **K-Nearest Neighbors (KNN)**. Berikut penjelasan tahapan pemodelan serta parameter yang digunakan:

### 1. Random Forest  
Random Forest adalah algoritma ensemble learning yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting.  
- **Parameter utama yang digunakan:** parameter default dari library `scikit-learn`.  
- **Kelebihan:**  
  - Tahan terhadap overfitting  
  - Mampu menangani data numerik dan multi-kelas dengan baik  
  - Memberikan estimasi pentingnya fitur (feature importance)  
- **Kekurangan:**  
  - Bisa jadi lambat pada dataset yang sangat besar  
  - Model kurang interpretatif dibanding model linear sederhana  

### 2. Support Vector Machine (SVM)  
SVM mencari hyperplane optimal untuk memisahkan kelas data. Kernel RBF digunakan untuk menangani pola non-linear.  
- **Parameter utama yang digunakan:** `kernel='rbf'` dengan parameter default lainnya.  
- **Kelebihan:**  
  - Efektif pada ruang berdimensi tinggi  
  - Dapat memodelkan pola non-linear  
- **Kekurangan:**  
  - Sensitif terhadap pemilihan kernel dan parameter  
  - Kurang efisien pada dataset sangat besar  

### 3. K-Nearest Neighbors (KNN)  
KNN mengklasifikasikan sampel berdasarkan mayoritas kelas dari k tetangga terdekat.  
- **Parameter utama yang digunakan:** `n_neighbors=5` (default).  
- **Kelebihan:**  
  - Algoritma sederhana dan intuitif  
  - Tidak memerlukan proses pelatihan eksplisit  
- **Kekurangan:**  
  - Performa menurun pada dataset besar  
  - Rentan terhadap fitur dengan skala berbeda sehingga perlu normalisasi

---

### Proses Pemodelan  
- Ketiga model dilatih menggunakan data train yang sudah diproses dan dinormalisasi.  
- Data dibagi dengan rasio 70% train dan 30% test menggunakan stratifikasi kelas untuk menjaga distribusi target tetap proporsional.  
- Setiap model diuji pada data uji dan dievaluasi menggunakan metrik akurasi, precision, recall, dan F1-score.  

---

### Pemilihan Model Terbaik  
Model terbaik dipilih berdasarkan perbandingan hasil metrik evaluasi pada data uji. Dari ketiga model yang diuji—Random Forest, Support Vector Machine (SVM), dan K-Nearest Neighbors (KNN)—Random Forest dipilih sebagai solusi utama.  

Alasan pemilihan Random Forest adalah karena model ini memberikan akurasi yang lebih tinggi dan performa lebih stabil khususnya dalam mendeteksi kelas *Hazardous*, yang sangat penting untuk pengambilan keputusan cepat dalam pengawasan kualitas udara. Selain itu, Random Forest memiliki kemampuan interpretasi fitur yang baik melalui fitur penting (feature importance), sehingga memudahkan analisis faktor lingkungan utama yang mempengaruhi kualitas udara.  

Kelebihan ini membuat Random Forest menjadi pilihan yang tepat untuk aplikasi klasifikasi kualitas udara di proyek ini.


---

## Evaluation

### Metrik Evaluasi yang Digunakan  
Dalam proyek klasifikasi kualitas udara ini, metrik evaluasi utama yang digunakan meliputi:

- **Akurasi (Accuracy)**  
  Mengukur proporsi prediksi yang benar dari seluruh data uji. Akurasi memberikan gambaran umum performa model dalam mengklasifikasikan semua kelas.  
  \[
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  \]

- **Precision**  
  Mengukur ketepatan prediksi positif untuk masing-masing kelas, yaitu proporsi data yang diprediksi sebagai kelas tertentu yang benar-benar termasuk kelas tersebut.  
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]

- **Recall (Sensitivity)**  
  Mengukur kemampuan model untuk menangkap seluruh data positif yang sebenarnya pada kelas tertentu.  
  \[
  \text{Recall} = \frac{TP}{TP + FN}
  \]

- **F1-Score**  
  Harmonik rata-rata dari precision dan recall, memberikan keseimbangan antara kedua metrik tersebut.  
  \[
  F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

Di mana TP = True Positive, FP = False Positive, FN = False Negative, dan TN = True Negative.

---

### Hasil Evaluasi Proyek

Berikut adalah hasil evaluasi model pada data uji menggunakan tiga algoritma yang diuji:

#### KNN Classification Report  
- **Akurasi:** 93.3%  
- **Precision:** Tinggi pada kelas *Good* (0.99) dan *Moderate* (0.92), lebih rendah pada kelas *Hazardous* (0.91) dan *Poor* (0.85).  
- **Recall:** Kelas *Good* dan *Moderate* sangat baik (>0.95), namun kelas *Hazardous* lebih rendah (0.76) yang menunjukkan model agak kesulitan mendeteksi semua kasus berbahaya.  
- **F1-Score:** Rata-rata tertimbang sekitar 0.93.

#### SVM Classification Report  
- **Akurasi:** 93.9%  
- **Precision & Recall:** Lebih seimbang dibanding KNN, dengan precision dan recall kelas *Hazardous* masing-masing 0.87 dan 0.81.  
- **F1-Score:** Rata-rata tertimbang meningkat menjadi 0.94, menunjukkan performa yang lebih stabil.

#### Random Forest Classification Report  
- **Akurasi:** 94.9% (terbaik dari ketiga model)  
- **Precision dan Recall:** Kelas *Hazardous* memiliki precision 0.88 dan recall 0.81, yang menandakan kemampuan model dalam mengenali kasus berbahaya cukup baik.  
- **F1-Score:** Rata-rata tertimbang sebesar 0.95, menunjukkan keseimbangan terbaik antara precision dan recall.

---

### Interpretasi Hasil

Model **Random Forest** menunjukkan performa terbaik dengan akurasi tertinggi dan metrik klasifikasi yang konsisten di seluruh kelas, terutama pada kelas kritis *Hazardous*. Hal ini mengindikasikan bahwa Random Forest mampu menangkap pola kompleks dalam data kuantitatif kualitas udara secara efektif.

Model SVM juga memberikan performa yang baik dan stabil, sementara KNN sedikit tertinggal terutama dalam hal recall kelas *Hazardous*, yang penting untuk deteksi dini kondisi berbahaya.

---

### Hubungan dengan Feature Importance

Analisis feature importance dari model Random Forest menguatkan hasil evaluasi, dimana fitur-fitur utama yang berkontribusi besar seperti **CO** (34.3%) dan **Proximity to Industrial Areas** (28.6%) menjadi indikator penting untuk klasifikasi kualitas udara. Fitur lain seperti NO2, SO2, dan Temperature juga memiliki peranan yang signifikan.

---

### Kesimpulan

Evaluasi menggunakan metrik akurasi, precision, recall, dan F1-score mengindikasikan bahwa model Random Forest adalah solusi paling efektif untuk klasifikasi kualitas udara pada dataset ini, dengan kemampuan terbaik dalam mengenali kondisi berbahaya (*Hazardous*), yang sangat penting bagi kesehatan masyarakat.

---

## Ringkasan Tabel Feature Importance

| Feature                   | Importance |
|---------------------------|------------|
| CO                        | 0.343      |
| Proximity to Industrial Areas | 0.286      |
| NO2                       | 0.096      |
| SO2                       | 0.091      |
| Temperature               | 0.072      |
| Population Density         | 0.040      |
| Humidity                  | 0.036      |
| PM10                      | 0.022      |
| PM2.5                     | 0.014      |

---

Dengan metrik evaluasi dan analisis fitur yang lengkap, proyek ini berhasil membangun model klasifikasi kualitas udara yang andal dan informatif.

