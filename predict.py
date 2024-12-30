import torch
import cv2
from torchvision import transforms
from facenet_pytorch import MTCNN

emotions = ['Гнев', 'Презрение', 'Отвращение', 'Страх', 'Радость', 'Нейтральное', 'Печаль', 'Удивление']

def preprocess_image(face):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(face).unsqueeze(0)

def predict_emotion(image_path, model):
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detector = MTCNN(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        boxes, probs = detector.detect(image_rgb)

        if boxes is None or boxes.size == 0:
            print("Лицо не найдено.")
            return "Ошибка при анализе фото. Попробуйте снова!"

        x1, y1, x2, y2 = boxes[0].astype(int)
        face = image_rgb[y1:y2, x1:x2]

        print(f"Размеры лица: {face.shape}")

        face_tensor = preprocess_image(face)

        print(f"Форма тензора: {face_tensor.shape}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        face_tensor = face_tensor.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(face_tensor)
            print(f"Выход модели: {outputs}")

            _, predicted_index = torch.max(outputs, 1)
            predicted_index = predicted_index.item()

        emotion = emotions[predicted_index]
        print(f"Распознанная эмоция: {emotion}")
        return f"{emotion}"
    except Exception as e:
        print(f"Ошибка: {e}")
        return "Ошибка при анализе фото. Попробуйте снова!"