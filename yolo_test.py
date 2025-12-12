from ultralytics import YOLO
from ultralytics import settings
import torch
from torchvision.ops import box_iou

# Example of IoU calculation boxes in (x1,y1,x2,y2) format
boxes1 = torch.tensor([[10, 10, 20, 20], [0, 0, 5, 5]], dtype=torch.float)
boxes2 = torch.tensor([[15, 15, 30, 30]], dtype=torch.float)

iou = box_iou(boxes1, boxes2)  # shape (2,1)
print(iou)

print(f"Using torch version: {torch.__version__}")
print(f'CUDA is available: {torch.cuda.is_available()}')

# actual code

images_directory = './data/KITTI_Selection/images'

# Load a pre-trained model
model = YOLO("yolov8n.pt")

# Perform inference on an image
results = model(images_directory, device='cuda:0')

print('Displaying results...')

# Show results
for result in results:
    file_name = str(result.path).split('/')[-1]
    print('From file: ', file_name)  # access image filename
    result.show()
