import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import sys
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

try:
    from chart_class_model import SimpleChartCNN as OriginalSimpleChartCNN, get_data_transform as original_transform
except ImportError:
    print("Lỗi: Không tìm thấy file 'chart_class_model.py'.")
    sys.exit(1)

# --- [1] CÁC THAM SỐ TINH CHỈNH ---
# (Dành cho bộ dữ liệu 2545 ảnh, ưu tiên độ chính xác)

GRAPH_DIR = './graphs'
SAVED_MODEL_PATH = 'chart_classifier_model.pth'

# 1. Tỷ lệ chia dữ liệu (Giữ nguyên, 70:20:10 là tốt)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# 2. Tham số Huấn luyện
IMAGE_SIZE = 128
BATCH_SIZE = 32           # Tăng lên 32 để huấn luyện ổn định hơn
LEARNING_RATE = 0.001     # 1e-3 là điểm khởi đầu tốt cho AdamW
WEIGHT_DECAY = 1e-5       # Thêm một chút điều chuẩn (regularization)
NUM_EPOCHS = 100          # Đặt số epoch cao, Early Stopping sẽ lo phần còn lại

# 3. Tham số cho Scheduler và Early Stopping
SCHEDULER_PATIENCE = 3    # Giảm LR nếu Val Loss không cải thiện sau 5 epochs
EARLY_STOPPING_PATIENCE = 5  # Dừng hẳn nếu Val Loss không cải thiện sau 15 epochs


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
        
        dummy_input = torch.randn(1, in_channels, IMAGE_SIZE, IMAGE_SIZE)
        linear_input_size = self._get_conv_output_dim(dummy_input)
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output_dim(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ChartDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, file_name)
        
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Lỗi khi tải ảnh: {img_path}. Lỗi: {e}")
            return None, None

        if file_name.startswith('bar'):
            label = 0
        elif file_name.startswith('pie'):
            label = 1
        else:
            label = -1
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return train_transform, val_test_transform

def run_training():
    if not os.path.exists(GRAPH_DIR):
        raise FileNotFoundError(f"Lỗi: Thư mục '{GRAPH_DIR}' không tồn tại. Vui lòng tạo thư mục và thêm ảnh.")

    all_files = [f for f in os.listdir(GRAPH_DIR) if f.endswith(('.jpg', '.png', '.jpeg')) 
                 and (f.startswith('bar') or f.startswith('pie'))]
    
    if len(all_files) == 0:
        raise ValueError(f"Không tìm thấy file ảnh 'bar*.jpg' hoặc 'pie*.jpg' trong '{GRAPH_DIR}'.")

    # Print class distribution (helpful to detect imbalance)
    from collections import Counter
    labels = []
    for f in all_files:
        if f.startswith('bar'):
            labels.append('bar')
        elif f.startswith('pie'):
            labels.append('pie')
        else:
            labels.append('other')
    counter = Counter(labels)
    total = len(all_files)
    print("Class distribution:")
    for k, v in counter.items():
        print(f"  {k}: {v} ({v/total:.2%})")

    # Stratified split to preserve class ratio across train/val/test
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = ['bar' if f.startswith('bar') else 'pie' if f.startswith('pie') else 'other' for f in all_files]
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=42)
        for train_val_idx, test_idx in sss.split(all_files, labels):
            train_val_files = [all_files[i] for i in train_val_idx]
            test_files = [all_files[i] for i in test_idx]

        val_fraction = VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)
        labels_trainval = [labels[i] for i in train_val_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
        for train_idx, val_idx in sss2.split(train_val_files, labels_trainval):
            train_files = [train_val_files[i] for i in train_idx]
            val_files = [train_val_files[i] for i in val_idx]
    except Exception as e:
        # Fallback to simple stratified split (older approach) or non-stratified if necessary
        print(f"Warning: stratified split failed ({e}), falling back to train_test_split.")
        try:
            train_val_files, test_files = train_test_split(
                all_files,
                test_size=TEST_RATIO,
                random_state=42,
                stratify=labels
            )
            train_files, val_files = train_test_split(
                train_val_files,
                test_size=(VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)),
                random_state=42,
                stratify=[('bar' if f.startswith('bar') else 'pie' if f.startswith('pie') else 'other') for f in train_val_files]
            )
        except Exception as e2:
            print(f"Fallback stratified split failed ({e2}), using non-stratified random splits.")
            train_val_files, test_files = train_test_split(all_files, test_size=TEST_RATIO, random_state=42)
            train_files, val_files = train_test_split(train_val_files, test_size=(VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)), random_state=42)

    # print class distribution per split for verification
    from collections import Counter
    def _print_split(name, files):
        cnt = Counter('bar' if f.startswith('bar') else 'pie' if f.startswith('pie') else 'other' for f in files)
        total = len(files)
        print(f"{name}: {total} items")
        for k, v in cnt.items():
            print(f"  {k}: {v} ({v/total:.2%})")

    _print_split('Train', train_files)
    _print_split('Validation', val_files)
    _print_split('Test', test_files)

    print(f"Tổng số ảnh: {len(all_files)}")
    print(f"Tập huấn luyện (Train): {len(train_files)} ảnh (Đã bật Augmentation)")
    print(f"Tập kiểm định (Validation): {len(val_files)} ảnh")
    print(f"Tập kiểm tra (Test): {len(test_files)} ảnh")

    train_transform, val_test_transform = get_data_transforms()

    train_dataset = ChartDataset(GRAPH_DIR, train_files, transform=train_transform)
    val_dataset = ChartDataset(GRAPH_DIR, val_files, transform=val_test_transform)
    test_dataset = ChartDataset(GRAPH_DIR, test_files, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model = SimpleChartCNN(in_channels=1, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE)

    print(f"Bắt đầu huấn luyện cho tối đa {NUM_EPOCHS} epochs...")
    print(f"Early Stopping sẽ kích hoạt sau {EARLY_STOPPING_PATIENCE} epochs không cải thiện.")
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            if inputs.nelement() == 0: continue
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.nelement() == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print(f"Đã lưu mô hình tốt nhất tại {SAVED_MODEL_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Val loss không cải thiện. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping kích hoạt tại epoch {epoch+1}.")
            break
        
        scheduler.step(val_loss)

    print("Huấn luyện hoàn tất.")

    # Training finished; best model was saved during training loop
    print("\nĐang đánh giá trên tập Test... (gọi evaluate_saved_model nếu cần)")


def evaluate_model(model, dataloader, device, criterion, class_names=None):
    """Evaluate a PyTorch model on a dataloader and return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs.nelement() == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            batches += 1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    results = {}
    if len(all_labels) == 0:
        return {"error": "No samples in dataloader"}

    avg_loss = total_loss / max(1, batches)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=sorted(list(set(all_labels))))
    cm = confusion_matrix(all_labels, all_preds)

    results['loss'] = avg_loss
    results['accuracy'] = acc
    results['confusion_matrix'] = cm.tolist()
    classes = [str(c) for c in sorted(list(set(all_labels)))]
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    results['per_class'] = per_class

    if class_names is not None:
        mapped = {}
        for i, cls in enumerate(sorted(list(set(all_labels)))):
            name = class_names[cls] if cls < len(class_names) else str(cls)
            mapped[name] = per_class[str(cls)]
        results['per_class_named'] = mapped

    return results


def _split_files_for_eval(all_files):
    """Return train, val, test splits using same logic as run_training."""
    train_val_files, test_files = train_test_split(
        all_files,
        test_size=TEST_RATIO,
        random_state=42,
        stratify=[f[0] for f in all_files]
    )
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=(VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)),
        random_state=42,
        stratify=[f[0] for f in train_val_files]
    )
    return train_files, val_files, test_files


def evaluate_saved_model(model_path=SAVED_MODEL_PATH, root_dir=GRAPH_DIR, batch_size=BATCH_SIZE, class_names=None):
    """Load a saved model and evaluate it on the test split of `root_dir`."""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Lỗi: Thư mục '{root_dir}' không tồn tại.")

    all_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and (f.startswith('bar') or f.startswith('pie'))]
    if not all_files:
        raise ValueError("Không tìm thấy ảnh để đánh giá.")

    _, _, test_files = _split_files_for_eval(all_files)

    _, val_test_transform = get_data_transforms()
    test_dataset = ChartDataset(root_dir, test_files, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleChartCNN(in_channels=1, num_classes=2).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(model, test_loader, device, criterion, class_names=class_names)
    return metrics

if __name__ == '__main__':
    import json
    try:
        # Usage:
        #  python train_model_class.py          -> runs training
        #  python train_model_class.py eval     -> evaluates saved model and prints JSON
        #  python train_model_class.py eval <model_path>
        # if len(sys.argv) >= 2 and sys.argv[1].lower() in ('eval', 'evaluate'):
            model_path = sys.argv[2] if len(sys.argv) >= 3 else SAVED_MODEL_PATH
            metrics = evaluate_saved_model(model_path=model_path, root_dir=GRAPH_DIR, batch_size=BATCH_SIZE, class_names=['bar', 'pie'])
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        # else:
            # run_training()
    except (FileNotFoundError, ValueError, IOError) as e:
        print(e)