# ============================================================
# Homework (3) - FC size and dropout tuning
# Compare small/medium/large FC models and dropout 0 / 0.3 / 0.5
# Change one parameter at a time
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ============================================================
# Device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Dataset (CIFAR-10)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False)

# ============================================================
# Feature extractor (shared and fixed)
# ============================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16x16
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 8x8
        )
        # Match the course design: 256 channels with a 5x5 feature map.
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),  # Force output to 5x5 => 256*5*5 = 6400
        )

    def forward(self, x):
        return self.conv2(x).view(x.size(0), -1)  # [B, 6400]

# ============================================================
# Three FC model sizes
# ============================================================
class AlexNetFC(nn.Module):
    def __init__(self, fc_type="small", dropout=0.5):
        super().__init__()
        self.features = FeatureExtractor()
        fc_in = 256 * 5 * 5  # 6400

        if fc_type == "small":
            # Small FC model
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 10)
            )
        elif fc_type == "medium":
            # Medium FC model
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 10)
            )
        elif fc_type == "large":
            # Large FC model
            self.fc = nn.Sequential(
                nn.Linear(fc_in, 2048),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 10)
            )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

# ============================================================
# Training function
# ============================================================
def train_and_eval(fc_type, dropout, num_epochs=10, tag="exp"):
    model = AlexNetFC(fc_type=fc_type, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir=f"runs/{tag}")

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_loss  = running_loss / len(trainloader)
        train_acc = correct_train / total_train
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

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

        writer.add_scalar("Loss/train",      avg_loss,  epoch)
        writer.add_scalar("Accuracy/train",  train_acc, epoch)
        writer.add_scalar("Accuracy/test",   test_acc,  epoch)
        print(f"[{tag}] Epoch {epoch+1}/{num_epochs}  "
              f"Loss: {avg_loss:.4f}  Train: {train_acc:.4f}  Test: {test_acc:.4f}")

    writer.close()
    return train_losses, train_accs, test_accs

# ============================================================
# Experiment A: compare FC sizes with dropout fixed at 0.5
# ============================================================
print("\n" + "="*60)
print("Experiment A: FC Size Comparison (dropout fixed at 0.5)")
print("="*60)

fc_experiments = {
    "FC_small_dp0.5":  ("small",  0.5),
    "FC_medium_dp0.5": ("medium", 0.5),
    "FC_large_dp0.5":  ("large",  0.5),
}

results_fc = {}
for tag, (fc_type, dp) in fc_experiments.items():
    print(f"\n>>> FC type: {fc_type}, Dropout: {dp}")
    losses, train_accs, test_accs = train_and_eval(fc_type, dp, num_epochs=10, tag=tag)
    results_fc[tag] = {"losses": losses, "train_accs": train_accs, "test_accs": test_accs}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for tag, data in results_fc.items():
    label = tag.replace("_dp0.5", "")
    axes[0].plot(data["losses"],     label=label)
    axes[1].plot(data["train_accs"], label=label)
    axes[2].plot(data["test_accs"],  label=label)
for ax, title in zip(axes, ["Train Loss", "Train Accuracy", "Test Accuracy"]):
    ax.set_title(f"{title} (Different FC Sizes)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw3_fc_size_comparison.png", dpi=150)
plt.show()
print("\nFigure saved as hw3_fc_size_comparison.png")

# ============================================================
# Experiment B: compare dropout rates with FC fixed at small
# ============================================================
print("\n" + "="*60)
print("Experiment B: Dropout Comparison (FC fixed at small)")
print("="*60)

dp_experiments = {
    "FC_small_dp0":   ("small", 0.0),
    "FC_small_dp0.3": ("small", 0.3),
    "FC_small_dp0.5": ("small", 0.5),
}

results_dp = {}
for tag, (fc_type, dp) in dp_experiments.items():
    print(f"\n>>> FC type: {fc_type}, Dropout: {dp}")
    losses, train_accs, test_accs = train_and_eval(fc_type, dp, num_epochs=10, tag=tag)
    results_dp[tag] = {"losses": losses, "train_accs": train_accs, "test_accs": test_accs}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for tag, data in results_dp.items():
    label = tag.replace("FC_small_", "")
    axes[0].plot(data["losses"],     label=label)
    axes[1].plot(data["train_accs"], label=label)
    axes[2].plot(data["test_accs"],  label=label)
for ax, title in zip(axes, ["Train Loss", "Train Accuracy", "Test Accuracy"]):
    ax.set_title(f"{title} (Different Dropout Rates)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hw3_dropout_comparison.png", dpi=150)
plt.show()
print("Figure saved as hw3_dropout_comparison.png")

# ============================================================
# Overfitting diagnosis
# ============================================================
print("\n" + "="*60)
print("Overfitting Diagnosis (last epoch)")
print("="*60)
for tag, data in results_dp.items():
    tr = data["train_accs"][-1]
    te = data["test_accs"][-1]
    gap = tr - te
    status = "Possible overfitting, consider increasing dropout" if gap > 0.1 else "Training looks stable"
    print(f"{tag:25s}  Train: {tr:.4f}  Test: {te:.4f}  Gap: {gap:.4f}  {status}")

print("\nRun tensorboard --logdir=runs to open TensorBoard.")