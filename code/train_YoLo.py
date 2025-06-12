from ultralytics import YOLO

# Load a model
# model = YOLO("myyolov8n.yaml")  # build a new model from scratch
model = YOLO("/mnt/zlc/YoLo/ultralytics-main/yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(device=0,
            data="/mnt/zlc/YoLo/ultralytics-main/ultralytics/cfg/datasets/mycoco128.yaml", 
            task='detect',
            batch=16,epochs=1600,imgsz=1280,scale=1.0,degrees=90.0,  
            patience=300,
            fliplr=0.5,
            flipud=0.5)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format 