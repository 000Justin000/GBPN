import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library
images = datasets.ImageFolder("datasets/cats_dogs/", transform=transformations)
data_loader = torch.utils.data.DataLoader(images, batch_size=256, shuffle=True)

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.densenet121(pretrained=True).to(device)
# model = models.resnet18(pretrained=True).to(device)
model = models.alexnet(pretrained=True).to(device)

cat_image_list, dog_image_list = [], []

for inputs, labels in data_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    cat_image_list.append(outputs[labels == 0])
    dog_image_list.append(outputs[labels == 1])

cat_images = torch.cat(cat_image_list, dim=0)
dog_images = torch.cat(dog_image_list, dim=0)

torch.save(cat_images, 'datasets/cats_dogs/cat_images.pt')
torch.save(dog_images, 'datasets/cats_dogs/dog_images.pt')