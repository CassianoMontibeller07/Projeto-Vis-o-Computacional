import cv2
import os
from ultralytics import YOLO

# Configurações do modelo e do vídeo
model_path = r'C:\Users\lchri\Desktop\notebooK_VC\runs\detect\train\weights\best.pt'
video_path = r'C:\Users\lchri\Desktop\notebooK_VC\video_peito_frango.mp4'  # Substitua pelo caminho do vídeo real
output_folder = 'output_2_50epochs'
confidence_threshold = 0.1

# Carregar o modelo
model = YOLO(model_path)

# Criar a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Função para salvar imagens com a bounding box e organizar por classe
def save_detection_with_bbox(frame, detections, folder, frame_count):
    frame_height, frame_width = frame.shape[:2]

    # Iterar sobre as detecções
    for i in range(len(detections)):
        bbox = detections.xyxy[i].cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
        class_id = int(detections.cls[i].cpu().numpy())  # ID da classe
        class_name = model.names[class_id]  # Nome da classe

        # Criar diretório para a classe, se não existir
        class_folder = os.path.join(folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        # Desenhar a bounding box no frame
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)  # Cor verde para a bounding box
        thickness = 2  # Espessura da linha da caixa
        frame_with_bbox = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), color, thickness)

        # Adicionar o nome da classe
        confidence = detections.conf[i].item()  # Convertendo para float
        label = f"{class_name}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_with_bbox = cv2.putText(frame_with_bbox, label, (x1, y1 - 10), font, 0.6, color, 2, cv2.LINE_AA)

        # Salvar a imagem com a bounding box na pasta da classe
        filename = f"frame_{frame_count}_det_{i}.jpg"
        filepath = os.path.join(class_folder, filename)
        cv2.imwrite(filepath, frame_with_bbox)

# Inicializar o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro: Não foi possível abrir o vídeo. Verifique o caminho.")
    exit()

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=confidence_threshold)
    detections = results[0].boxes if results[0].boxes is not None else None

    if detections:
        save_detection_with_bbox(frame, detections, output_folder, frame_count)

    frame_count += 1

cap.release()
print(f"Processamento concluído. Frames processados: {frame_count}")
