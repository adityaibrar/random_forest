# Import library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Mengimpor dataset
datasets = pd.read_csv('Social_Network_Ads.csv')

# Memisahkan atribut (X) dan label (Y)
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Penskalaan fitur menggunakan StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Melatih model menggunakan RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

# Memprediksi hasil data uji
Y_pred = classifier.predict(X_test)

# Membuat Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Menampilkan heatmap dari Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Prediksi')
plt.ylabel('Sebenarnya')
plt.title('Confusion Matrix')
plt.show()

# Fungsi untuk visualisasi hasil data latih dan data uji
def plot_results(X_set, Y_set, title):
    # Membuat meshgrid untuk plotting
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    
    # Mewarnai area plot sesuai hasil prediksi model
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(['#FF0000', '#00FF00']))
    
    # Menampilkan scatter plot untuk setiap kelas
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(Y_set)):
        plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                    c=[['#FF0000', '#00FF00'][i]], label=j)
    
    # Menambahkan label dan judul plot
    plt.title(title)
    plt.xlabel('Usia')
    plt.ylabel('Estimasi Gaji')
    plt.legend()
    plt.show()

# Visualisasi hasil data latih
plot_results(X_train, Y_train, 'Random Forest Classifier (Data Latih)')

# Visualisasi hasil data uji
plot_results(X_test, Y_test, 'Random Forest Classifier (Data Uji)')
