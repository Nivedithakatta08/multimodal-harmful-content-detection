from torchvision.datasets import CIFAR10

print("Downloading...")

dataset = CIFAR10(
    root="data",
    train=True,
    download=True
)

print("Done")
print("Samples:", len(dataset))

image, label = dataset[0]

print(type(image))
print(label)