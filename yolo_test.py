from ultralytics import YOLO
from ultralytics import settings
import torch
from torchvision.ops import box_iou
import pandas as pd

# Example of IoU calculation boxes in (x1,y1,x2,y2) format
boxes1 = torch.tensor([[10, 10, 20, 20], [0, 0, 5, 5]], dtype=torch.float)
boxes2 = torch.tensor([[15, 15, 30, 30], [2, 2, 4, 4]], dtype=torch.float)

iou = box_iou(boxes1, boxes2)  # shape (2,1)
print(iou)

# print(f"Using torch version: {torch.__version__}")
print(f'CUDA is available: {torch.cuda.is_available()}')
# print(f'CUDA version: {torch.version.cuda}')

# exit(0)

# actual code
data_directory = r'./data/KITTI_Selection'
images_directory = data_directory + r'/images'
annotations_directory = data_directory + r'/labels'

# Load a pre-trained model
model = YOLO("yolov8n.pt")

# Create a reverse dictionary for class names
name_to_id = {v: k for k, v in model.model.names.items()}
car_class_id = name_to_id['car']
print(f'Class ID for "car": {car_class_id}')

# Perform inference on an image
results = model(images_directory, device='cuda:0', classes=[car_class_id])

print('Displaying results...')

# Show results
for result in results:
    result = result.to('cpu')  # move result to CPU for further processing
    frame_file_name = str(result.path).split('/')[-1]
    print('From file: ', frame_file_name)  # access image filename
    # print('Confidences: ', result.boxes.conf)  # numpy array of confidences
    print('Boxes: ', result.boxes.xyxy.shape[0])  # Boxes object for bbox outputs
    labels_file_name = annotations_directory + '/' + frame_file_name.replace('.png', '.txt')
    try:
        labels_df = pd.read_csv(labels_file_name, sep=' ', header=None)
    except pd.errors.EmptyDataError:
        labels_df = pd.DataFrame()
    # print(labels_df.head(2))
    labels_tensor = torch.tensor(labels_df.iloc[:, 1:5].values)
    print('Ground truth boxes: ', labels_tensor.shape[0])
    for box in result.boxes.xyxy:
        # print('Predicted box: ', box)
        iou_value = box_iou(labels_tensor, box.unsqueeze(0))
        iou_value = iou_value.squeeze()
        # print(f'IoU with detection box: {iou_value}')

    # result.show()  # display image in a window
