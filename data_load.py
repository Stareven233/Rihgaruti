from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader as DataLoader

root = './data/train'
img_size = 128
batch_size = 16
num_workers = 0

transform = transforms.Compose([
    transforms.Resize(img_size),  # 保持比例，将短边放缩为IMG_SIZE
    transforms.CenterCrop(img_size),  # 裁去长边多余部分，保证图片为不变形的长方形
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=root, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
