from ultralytics import YOLO

model= YOLO('yolov8x')

result = model.predict('dataset/pic/2.png', save= True)
# result = model.predict('dataset/matchrec.mp4', save=True)

print(result)
print('boxes:')

for box in result[0].boxes:
    print(box)
