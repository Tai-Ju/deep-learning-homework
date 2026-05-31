# ============================================================
# Homework (2) - AlexNet_Build hyperparameter tuning
# Tune learning rate and channels, then compare loss curves
# Change one parameter at a time
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Dataset (CIFAR-10, 32x32 RGB, 10 classes)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False)

# ============================================================
# Simplified AlexNet for 32x32 inputs
# ============================================================
class AlexNetSmall(nn.Module):
    def __init__(self, num_channels=6, fc_size=512, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=5, padding=2),  # tunable channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # 16x16
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # 8x8
        )
        fc_in = num_channels * 2 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================
# Training function
# ============================================================
def train_and_eval(lr, num_channels, fc_size=512, dropout=0.5, num_epochs=10, tag="exp"):
    model = AlexNetSmall(
        num_channels=num_channels,
        fc_size=fc_size,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=f"runs/{tag}")

    train_losses, test_accs = [], []

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)

        # --- Test ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        test_accs.append(test_acc)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        print(f"[{tag}] Epoch {epoch+1}/{num_epochs}  Loss: {avg_loss:.4f}  Test Acc: {test_acc:.4f}")

    writer.close()
    return train_losses, test_accs

# ============================================================
# Experiment A: learning rate tuning with channels fixed at 6
# ============================================================
print("\n" + "="*60)
print("Experiment A: Learning Rate Tuning (channels fixed at 6)")
print("="*60)

lr_experiments = {
    "lr_0.1":   0.1,
    "lr_0.01":  0.01,
    "lr_0.001": 0.001,
    # "lr_0.0001": 0.0001,  # uncomment to include this setting
}

results_lr = {}
for tag, lr in lr_experiments.items():
    print(f"\n>>> Learning rate = {lr}")
    losses, accs = train_and_eval(
        lr=lr, num_channels=6, fc_size=512, dropout=0.5, num_epochs=10, tag=tag
    )
    results_lr[tag] = {"losses": losses, "accs": accs}

# Plot learning rate experiment
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, data in results_lr.items():
    axes[0].plot(data["losses"], label=tag)
    axes[1].plot(data["accs"],   label=tag)
axes[0].set_title("Train Loss (Different Learning Rates)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Test Accuracy (Different Learning Rates)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw2_learning_rate_comparison.png", dpi=150)
plt.show()
print("\nFigure saved as hw2_learning_rate_comparison.png")

# ============================================================
# Experiment B: channel tuning with learning rate fixed at 0.001
# ============================================================
print("\n" + "="*60)
print("Experiment B: Channel Tuning (learning rate fixed at 0.001)")
print("="*60)

ch_experiments = {
    "ch_6":  6,
    "ch_16": 16,
    "ch_32": 32,
}

results_ch = {}
for tag, ch in ch_experiments.items():
    print(f"\n>>> Channels = {ch}")
    losses, accs = train_and_eval(
        lr=0.001, num_channels=ch, fc_size=512, dropout=0.5, num_epochs=10, tag=tag
    )
    results_ch[tag] = {"losses": losses, "accs": accs}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, data in results_ch.items():
    axes[0].plot(data["losses"], label=tag)
    axes[1].plot(data["accs"],   label=tag)
axes[0].set_title("Train Loss (Different Channel Sizes)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Test Accuracy (Different Channel Sizes)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw2_channel_comparison.png", dpi=150)
plt.show()
print("Figure saved as hw2_channel_comparison.png")

# ============================================================
# Experiment C: fully connected layer size tuning
# ============================================================
print("\n" + "="*60)
print("Experiment C: FC Size Tuning (lr fixed at 0.001, channels fixed at 16)")
print("="*60)

fc_experiments = {
    "fc_128": 128,
    "fc_256": 256,
    "fc_512": 512,
}

results_fc = {}
for tag, fc_size in fc_experiments.items():
    print(f"\n>>> FC size = {fc_size}")
    losses, accs = train_and_eval(
        lr=0.001, num_channels=16, fc_size=fc_size, dropout=0.5, num_epochs=10, tag=tag
    )
    results_fc[tag] = {"losses": losses, "accs": accs}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, data in results_fc.items():
    axes[0].plot(data["losses"], label=tag)
    axes[1].plot(data["accs"],   label=tag)
axes[0].set_title("Train Loss (Different FC Sizes)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Test Accuracy (Different FC Sizes)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw2_fc_size_comparison.png", dpi=150)
plt.show()
print("Figure saved as hw2_fc_size_comparison.png")

# ============================================================
# Experiment D: dropout tuning
# ============================================================
print("\n" + "="*60)
print("Experiment D: Dropout Tuning (lr fixed at 0.001, channels fixed at 16, FC size fixed at 512)")
print("="*60)

dropout_experiments = {
    "dropout_0.3": 0.3,
    "dropout_0.5": 0.5,
    "dropout_0.7": 0.7,
}

results_dropout = {}
for tag, dropout in dropout_experiments.items():
    print(f"\n>>> Dropout = {dropout}")
    losses, accs = train_and_eval(
        lr=0.001, num_channels=16, fc_size=512, dropout=dropout, num_epochs=10, tag=tag
    )
    results_dropout[tag] = {"losses": losses, "accs": accs}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, data in results_dropout.items():
    axes[0].plot(data["losses"], label=tag)
    axes[1].plot(data["accs"],   label=tag)
axes[0].set_title("Train Loss (Different Dropout Rates)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].set_title("Test Accuracy (Different Dropout Rates)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw2_dropout_comparison.png", dpi=150)
plt.show()
print("Figure saved as hw2_dropout_comparison.png")

print("\nRun tensorboard --logdir=runs to open TensorBoard.")
