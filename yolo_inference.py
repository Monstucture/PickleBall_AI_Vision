from ultralytics import YOLO

# model= YOLO('models/yolov5_last.pt')
model= YOLO('yolov8x.pt')

# result = model.predict('dataset/matchrec.mp4',conf=0.2, save= True)
result = model.track('dataset/matchrec.mp4', conf=0.2, save=True, save_txt=True)

# print(result)
# print('boxes:')

# for box in result[0].boxes:
#     print(box)
