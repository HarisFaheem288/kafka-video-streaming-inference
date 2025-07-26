import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

class ImageClassifier:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        with open("imagenet_classes.txt") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img_t)
        _, index = torch.max(out, 1)
        return self.classes[index]
