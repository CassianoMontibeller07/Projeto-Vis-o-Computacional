import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Caminhos das pastas e CSVs
train_csv_path = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\train\_classes.csv'
val_csv_path = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\valid\_classes.csv'
test_csv_path = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\test\_classes.csv'



# Caminho das pastas que contém as imagens (train, val, test)
train_image_folder = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\train'
val_image_folder = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\val'
test_image_folder = r'C:\Users\lchri\Desktop\Chicken Classification.v2i.multiclass (1)\test'

# Pasta de saída onde as imagens organizadas serão armazenadas
output_folder = 'yolo_classification_dataset'

# Função para organizar as imagens por classe
def organize_data(csv_path, image_folder, subset_name):
    # Carregar o CSV
    data = pd.read_csv(csv_path)
    
    # Organizar as imagens por classe
    for _, row in data.iterrows():
        image_name = row['image']
        image_path = os.path.join(image_folder, image_name)  # Caminho da imagem
        labels = row['labels'].split(',')  # Labels separados por vírgula

        # Para cada label, movemos a imagem para a pasta correspondente
        for label in labels:
            label_folder = os.path.join(output_folder, subset_name, label.strip())
            os.makedirs(label_folder, exist_ok=True)  # Cria a pasta da classe se não existir
            shutil.copy(image_path, os.path.join(label_folder, image_name))  # Copia a imagem para a classe

# Criar pasta de saída
os.makedirs(output_folder, exist_ok=True)

# Organizar os dados para treino, validação e teste
organize_data(train_csv_path, train_image_folder, 'train')
organize_data(val_csv_path, val_image_folder, 'val')
organize_data(test_csv_path, test_image_folder, 'test')

print("Dataset organizado com sucesso!")
