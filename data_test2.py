import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision
from torch import tensor

image =  Image.new("RGB", (448, 448), (0, 0, 0))


history_image_transformer = Compose(
    [
        Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
               max_size=None, antialias=True),
        CenterCrop(size=(224, 224)),
        ToTensor(),
        Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
    ]
)

image2 = torch.zeros(1, 3, 384, 384)

image = history_image_transformer(image)
image2 = history_image_transformer(image2)
print(image.shape)
print(image2.shape)