import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import settings
import torch
from torchvision.ops import box_iou
import pandas as pd
import numpy as np

print_info = True
if print_info:
    print("System Setup Information...")
    print(f"Using torch version: {torch.__version__}")
    print(f'CUDA is available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f"Ultralytics version: {settings.version}")


class Detection:
    def __init__(self, b_box, gt_box, distance, gt_distance, iou):
        self.b_box = b_box
        self.gt_box = gt_box
        self.distance = distance
        self.gt_distance = gt_distance
        self.iou = iou


# actual code
data_directory = r'./data/KITTI_Selection'
images_directory = data_directory + r'/images'
annotations_directory = data_directory + r'/labels'
cam_calib_directory = data_directory + r'/calib'

# Load a pre-trained model
model = YOLO("yolov8n.pt")

# Create a reverse dictionary for class names
name_to_id = {v: k for k, v in model.model.names.items()}
car_class_id = name_to_id['car']
print(f'Class ID for "car": {car_class_id}')

# Perform inference on an image
results = model(images_directory, device='cuda:0', classes=[car_class_id])

print('Displaying results...')

detections = []

# Show results
for result in results:
    matched_labels = []
    result = result.to('cpu')  # move result to CPU for further processing
    frame_file_name = str(result.path).split('/')[-1]
    print('From file: ', frame_file_name)  # access image filename
    # print('Confidences: ', result.boxes.conf)  # numpy array of confidences
    print('Detection boxes: ', result.boxes.xyxy.shape[0])  # Boxes object for bbox outputs

    labels_file_name = annotations_directory + '/' + frame_file_name.replace('.png', '.txt')

    # result.show()  # display image in a window

    try:
        labels_df = pd.read_csv(labels_file_name, sep=' ', header=None)
    except pd.errors.EmptyDataError:
        # labels_df = pd.DataFrame()
        continue

    calbration_file_name = cam_calib_directory + '/' + frame_file_name.replace('.png', '.txt')
    camera_matrix_frame = np.loadtxt(calbration_file_name, delimiter=' ', dtype=np.float32)
    camera_matrix_frame = torch.tensor(camera_matrix_frame)
    # print('Camera matrix ', camera_matrix_frame)

    # print(labels_df.head(2))
    labels_tensor = torch.tensor(labels_df.iloc[:, 1:5].values)
    print('Ground truth boxes: ', labels_tensor.shape[0])
    print(labels_df.iloc[:, -1].values)

    for box in result.boxes.xyxy:
        # print('Predicted box: ', box)
        iou_values = box_iou(labels_tensor, box.unsqueeze(0)).squeeze()
        max_iou_value = torch.max(iou_values).item()
        max_iou_idx = torch.argmax(iou_values).item()
        # print('Max IoU: ', max_iou_value, ' at index ', max_iou_idx)

        if max_iou_value < 0.7 or labels_tensor.shape[0] == 0 or max_iou_idx in matched_labels:
            print('No matching ground truth box found, skipping distance calculation.')
            continue

        matched_labels.append(max_iou_idx)
        foot_point_x = (box[0] + box[2]) / 2
        foot_point_y = box[3]
        # print(f'Foot point: ({foot_point_x:.1f}, {foot_point_y:.1f})')
        foot_point = torch.tensor([[foot_point_x, foot_point_y, 1.0]])
        world_point = torch.matmul(torch.inverse(camera_matrix_frame), foot_point.T).squeeze()
        # print(f'World coordinates: X={world_point[0][0]:.2f}, Y={world_point[1][0]:.2f}, Z={world_point[2][0]:.2f}')

        scale = 1.65 / world_point[1]
        world_point_scaled = world_point * scale
        # print(
        # f'Estimated ground distance: {(np.sqrt(world_point_scaled[1][0]**2 + world_point_scaled[2][0]**2))} meters')

        detection = Detection(
            b_box=box.numpy(),
            gt_box=labels_tensor[torch.argmax(iou_values)].numpy(),
            distance=np.sqrt(world_point_scaled[1]**2 + world_point_scaled[2]**2),
            # distance=np.linalg.norm(world_point_scaled[:, 0].numpy()),
            gt_distance=labels_df.iloc[torch.argmax(iou_values).item(), -1],
            iou=max_iou_value
        )
        detections.append(detection)

        # difference = np.linalg.norm(world_point_scaled.numpy()) - detection.distance
        # print('Line of sight vs Ground distance:', difference / detection.gt_distance * 100, '%')

print(f'Total detections processed: {len(detections)}')

all_estimated_distances = [det.distance for det in detections]
all_gt_distances = [det.gt_distance for det in detections]


plt.scatter(all_estimated_distances, all_gt_distances, c='blue')
plt.plot([0, max(all_gt_distances)], [0, max(all_gt_distances)], 'r--')
plot_file_name = data_directory + r'/distance_estimation_scatter.png'
plt.savefig(plot_file_name)
plt.show()
