import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

IMAGE_SIZE = 128
LABELS_MAP = {0: 'Bar Chart', 1: 'Pie Chart'}

class SimpleChartCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SimpleChartCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_data_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def load_model_and_predict(image_path, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy mô hình đã lưu tại '{model_path}'.")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy ảnh tại '{image_path}'.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleChartCNN(in_channels=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    data_transform = get_data_transform()
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise IOError(f"Lỗi khi tải ảnh '{image_path}': {e}")

    tensor = data_transform(image)
    tensor = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        probs_list = probabilities.cpu().numpy().tolist()[0]
        prediction_idx = int(torch.argmax(probabilities, dim=1).item())
        # confidence as percentage for the predicted class
        confidence = probs_list[prediction_idx] * 100

        # Print per-class probabilities for debugging
        try:
            labels = [LABELS_MAP.get(i, str(i)) for i in range(len(probs_list))]
        except Exception:
            labels = [str(i) for i in range(len(probs_list))]
        probs_repr = ", ".join([f"{lab}: {p*100:.2f}%" for lab, p in zip(labels, probs_list)])
        print(f"Per-class probabilities -> {probs_repr}")
        
    prediction_label = LABELS_MAP.get(prediction_idx, 'Unknown')
    
    return prediction_label, confidence