import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import settings
import torch
from torchvision.ops import box_iou
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

print_info = False
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

# Perform inference on an image
results = model(images_directory, device='cuda:0', classes=[car_class_id])

print('Displaying results...')

detections = []

# Show results
for result in results:
    matched_labels = []
    result = result.to('cpu')  # move result to CPU for further processing
    frame_file_name = str(result.path).split('/')[-1]
    print('Processing frame: ', frame_file_name)  # access image filename
    # print('Confidences: ', result.boxes.conf)  # numpy array of confidences
    # print('Detection boxes: ', result.boxes.xyxy.shape[0])  # Boxes object for bbox outputs

    labels_file_name = annotations_directory + '/' + frame_file_name.replace('.png', '.txt')

    show_results = False
    if show_results:
        result.show()  # display image in a window

    try:
        labels_df = pd.read_csv(labels_file_name, sep=' ', header=None)
    except pd.errors.EmptyDataError:
        continue

    calbration_file_name = cam_calib_directory + '/' + frame_file_name.replace('.png', '.txt')
    camera_matrix_frame = np.loadtxt(calbration_file_name, delimiter=' ', dtype=np.float32)
    camera_matrix_frame = torch.tensor(camera_matrix_frame)

    labels_tensor = torch.tensor(labels_df.iloc[:, 1:5].values)

    # Visualize the labels
    image_path = images_directory + '/' + frame_file_name
    image_with_labels = Image.open(image_path)
    image_with_detections = image_with_labels.copy()
    draw_labels = ImageDraw.Draw(image_with_labels)
    draw_detections = ImageDraw.Draw(image_with_detections)

    for label_tensor in labels_tensor:
        draw_labels.rectangle(label_tensor.tolist(), outline='white', width=2)

    labels_file_name = data_directory + '/' + frame_file_name.split('.')[0] + '__labels.png'
    image_with_labels.save(labels_file_name)

    for box in result.boxes.xyxy:
        iou_values = box_iou(labels_tensor, box.unsqueeze(0)).squeeze()
        max_iou_value = torch.max(iou_values).item()
        max_iou_idx = torch.argmax(iou_values).item()

        if max_iou_value < 0.69 or labels_tensor.shape[0] == 0 or max_iou_idx in matched_labels:
            draw_detections.rectangle(box.tolist(), outline='yellow', width=2)
            draw_detections.text((box[2], box[3]), f'IoU: {max_iou_value:.2f}', fill='red', anchor='rb')
            continue

        draw_detections.rectangle(box.tolist(), outline='green', width=2)
        draw_detections.text((box[2], box[3]), f'IoU: {max_iou_value:.2f}', fill='red', anchor='rb')

        matched_labels.append(max_iou_idx)
        foot_point_x = (box[0] + box[2]) / 2
        foot_point_y = box[3]
        foot_point = torch.tensor([[foot_point_x, foot_point_y, 1.0]])
        world_point = torch.matmul(torch.inverse(camera_matrix_frame), foot_point.T).squeeze()

        scale = 1.65 / world_point[1]
        world_point_scaled = world_point * scale

        detection = Detection(
            b_box=box.numpy(),
            gt_box=labels_tensor[torch.argmax(iou_values)].numpy(),
            # X and Z coordinates only as we are in the XY-plane
            distance=np.linalg.norm(world_point_scaled[::2].numpy()),
            gt_distance=labels_df.iloc[torch.argmax(iou_values).item(), -1],
            iou=max_iou_value
        )
        detections.append(detection)

        distance_error = (detection.distance - detection.gt_distance) / detection.gt_distance

        # Visualize the detection
        image_path = images_directory + '/' + frame_file_name
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Draw predicted box in red
        draw.rectangle(detection.b_box.tolist(), outline='red', width=2)
        draw.text((detection.b_box[2], detection.b_box[3]),
                  f'Est: {detection.distance:.2f}m', fill='red', anchor='rb')
        # Draw ground truth box in green
        draw.rectangle(detection.gt_box.tolist(), outline='green', width=2)
        draw.text((detection.gt_box[0], detection.gt_box[1]), f'GT: {detection.gt_distance:.2f}m',
                  fill='green', anchor='lb')

        vis_file_name = data_directory + '/' + frame_file_name.split('.')[0] + '_' + str(max_iou_idx)

        if (np.abs(distance_error) > 0.2):
            vis_file_name += '_large_error.png'
            image.save(vis_file_name)
        else:
            vis_file_name += '.png'
            image.save(vis_file_name)

    frame_recall = len(matched_labels) / (labels_tensor.shape[0])
    frame_precision = len(matched_labels) / (result.boxes.xyxy.shape[0])
    draw_detections.text((10, 10), f'Precision: {frame_precision:.2f}', fill='red', anchor='lt')
    draw_detections.text((10, 30), f'Recall: {frame_recall:.2f}', fill='red', anchor='lt')
    detections_file_name = data_directory + '/' + frame_file_name.split('.')[0] + '__detections.png'
    image_with_detections.save(detections_file_name)


print(f'Total detections processed: {len(detections)}')

all_estimated_distances = [det.distance for det in detections]
all_gt_distances = [det.gt_distance for det in detections]


plt.scatter(all_estimated_distances, all_gt_distances, c='blue')
plt.plot([0, max(all_gt_distances)], [0, max(all_gt_distances)], 'r')
plt.plot([0, max(all_gt_distances)], [0, max(all_gt_distances)*0.8], 'r--')
plt.plot([0, max(all_gt_distances)], [0, max(all_gt_distances)*1.2], 'r--')
plt.xlabel('Estimated Distance (m)')
plt.ylabel('Ground Truth Distance (m)')
plt.title('Distance Estimation Evaluation')
plot_file_name = data_directory + r'/distance_estimation_scatter.png'
plt.savefig(plot_file_name)
# plt.show()

print('Evaluation completed.')
