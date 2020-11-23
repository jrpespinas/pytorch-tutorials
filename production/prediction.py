import json

from torchvision import models
from prepare import transform_image

model = models.densenet121(pretrained=True)
model.eval()

imagenet_class_index = json.load(open('imagenet_class_index.json'))


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


with open("cat.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
