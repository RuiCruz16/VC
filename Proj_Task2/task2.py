import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# ==============================================================================
# 1. CONFIGURATIONS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "dataset"
IMG_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 2 # Class 0: Background, Class 1: Ball

# ==============================================================================
# 2. DATASET & DATALOADER
# ==============================================================================
class PoolDataset(Dataset):
    """
    Custom Dataset to load YOLO formatted data into PyTorch Object Detection format.
    """
    def __init__(self, split="train"):
        self.img_dir = os.path.join(DATASET_ROOT, split, "images")
        self.lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
        
        # Keep only valid image files
        self.images = [f for f in sorted(os.listdir(self.img_dir)) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Load Image (OpenCV) ---
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Resize image to training size (512x512)
        rgb_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        
        # Convert to PyTorch Tensor (Channels, Height, Width) and normalize to [0, 1]
        img_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0

        # --- Load Annotations (YOLO format to Pascal VOC format) ---
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class cx cy w h (normalized 0 to 1)
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to absolute coordinates [x_min, y_min, x_max, y_max] for 512px
                        x1 = (cx - w / 2) * IMG_SIZE
                        y1 = (cy - h / 2) * IMG_SIZE
                        x2 = (cx + w / 2) * IMG_SIZE
                        y2 = (cy + h / 2) * IMG_SIZE
                        
                        # Clip boxes to image boundaries to prevent PyTorch crashes
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(IMG_SIZE - 1, x2), min(IMG_SIZE - 1, y2)
                        
                        # Ensure box is valid
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])

        # --- Build Target Dictionary ---
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            # PyTorch requires background=0. Therefore, all balls become class 1!
            labels_tensor = torch.ones((len(boxes),), dtype=torch.int64) 
        else:
            # Handle empty tables (0 balls)
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx])
        }

        return img_tensor, target

def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes per image in a batch.
    Required for PyTorch Object Detection DataLoaders.
    """
    return tuple(zip(*batch))

# ==============================================================================
# 3. MODELS (ARCHITECTURES)
# ==============================================================================
def build_faster_rcnn(num_classes=NUM_CLASSES):
    """
    Builds a Faster R-CNN model (Two-Stage Detector).
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # Replace the classification head to match our number of classes (2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def build_retinanet(num_classes=NUM_CLASSES):
    """
    Builds a RetinaNet model (Single-Stage Detector).
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weights)
    
    # Replace the classification head for RetinaNet
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    return model

# ==============================================================================
# 4. TRAINING ENGINE & EXECUTION
# ==============================================================================
def train_model(model, name, dataloader, epochs=5):
    """
    Standard PyTorch training loop for Object Detection models.
    """
    print(f"\n--- Starting training for {name} ---")
    model.to(DEVICE)
    
    # Optimizer: SGD is the standard choice for PyTorch detection models
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move images and targets to the selected device (CPU/GPU)
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward pass (in train mode, models return a dictionary of losses)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass (Calculate gradients and update weights)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {losses.item():.4f}")
                
        print(f">>> End of Epoch {epoch+1} | Average Loss: {epoch_loss/len(dataloader):.4f}")
        
    # Save the trained weights as requested in the PDF deliverables (.pth)
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/{name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    
    # 1. Initialize Dataset and DataLoader
    train_dataset = PoolDataset(split="train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # Keep at 0 to avoid multiprocessing errors on Windows
    )
    print(f"Loaded {len(train_dataset)} training images.")
    
    # 2. Build Models
    print("Building Faster R-CNN...")
    faster_rcnn = build_faster_rcnn()
    
    # print("Building RetinaNet...")
    # retinanet = build_retinanet()f
    
    # 3. Train! (We'll do just 1 epoch to test if the engine works)
    train_model(faster_rcnn, "faster_rcnn", train_loader, epochs=1)
    
    # train_model(retinanet, "retinanet", train_loader, epochs=1)
    
    print("\nSanity Check Complete!")
