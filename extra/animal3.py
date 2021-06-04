import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library
images = datasets.ImageFolder("datasets/animals/", transform=transformations)
data_loader = torch.utils.data.DataLoader(images, batch_size=30, shuffle=True)

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet(pretrained=True).to(device)
# model = models.resnet18(pretrained=True).to(device)
# model = models.densenet121(pretrained=True).to(device)

cat_image_list, dog_image_list, panda_image_list = [], [], []

for inputs, labels in data_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    cat_image_list.append(outputs[labels == 0])
    dog_image_list.append(outputs[labels == 1])
    panda_image_list.append(outputs[labels == 2])

cat_images = torch.cat(cat_image_list, dim=0)
dog_images = torch.cat(dog_image_list, dim=0)
panda_images = torch.cat(panda_image_list, dim=0)

torch.save(cat_images, 'datasets/animals/cat_images.pt')
torch.save(dog_images, 'datasets/animals/dog_images.pt')
torch.save(panda_images, 'datasets/animals/panda_images.pt')