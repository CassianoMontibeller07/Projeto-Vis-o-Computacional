# Ativação de ambiente virtual e instalação de pacotes
# Execute estes comandos manualmente no terminal:
# venv\Scripts\activate      # Ativa o ambiente virtual no Windows
# pip install ultralytics notebook jupyter albumentations opencv-python-headless matplotlib

# Importações
from ultralytics import YOLO, settings
import cv2
import albumentations as A
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

# Exibir o valor de GITHUB_ASSETS_STEMS
print("GITHUB_ASSETS_STEMS:", GITHUB_ASSETS_STEMS)

# Exibir as configurações do Ultralytics
print("YOLO Settings:", settings)

# Inicializar o modelo YOLOv8 Nano
model = YOLO('yolov8n')  # 'yolov8n' é o modelo nano, rápido e leve
print("Modelo YOLOv8 carregado com sucesso!")

# Caminho para o arquivo YAML do dataset
data_path = r"Teste.v2i.yolov8/data.yaml"

# Treinamento do modelo
# Treinamento do modelo
model.train(
    data=data_path,       # Caminho para o dataset
    epochs=50,            # Número de épocas (não muito alto para evitar overfitting)
    patience=10,          # Número de épocas para early stopping sem melhora
    batch=8,              # Tamanho do lote (menor devido ao dataset pequeno)
    imgsz=640,            # Tamanho das imagen            
    pretrained=False,     # Não usar pesos pré-treinados (se você estiver treinando do zero)
    resume=False,         # Não retomar treinamento anterior
    single_cls=False,     # Detectar várias classes
    box=7.5,              # Peso da perda para caixas delimitadoras
    cls=0.5,              # Peso da perda para classificações
    dfl=0.5,              # Foco dinâmico da perda
    val=True,             # Avaliar na validação após cada época
    lr0=0.001,            # Taxa de aprendizado inicial (ajustada para dataset pequeno)
    lrf=0.1,              # Taxa de aprendizado final
    weight_decay=0.0001,  # Regularização para evitar overfitting
    degrees=0.1,          # Aumento de rotação (ajuste leve)
    hsv_s=0.3,            # Alteração de saturação
    hsv_v=0.3,            # Alteração de brilho
    scale=0.5,            # Aumento de escalonamento
    #fliplr=0.5            # Aumento com flip horizontal
)



print("Treinamento concluído!")
