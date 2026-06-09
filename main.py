import torch
import torch.nn as nn
from frspecmuon import FrSpecMuon

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from riemann_layers import RiemannianLinear, riemannize
from r_optimizer import FrSpecMuon


# --------------------------------------------------
# Simple MNIST MLP
# --------------------------------------------------

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

        )

        self.head = nn.Linear(128, 10) 

    def forward(self, x):

        # flatten image
        x = x.view(x.size(0), -1)

        return self.head(self.body(x))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128
)


# --------------------------------------------------
# Model + optimizer
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MNISTNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters())

#Train normally for for 5 rounds
print("Normal pre-training:")
for epoch in range(5):
    model.train()

    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
       
        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1} "
        f"Train Loss: {avg_loss:.4f}"
    )





model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = 100.0 * correct / total

print("Model evaluation after normal epochs:")
print(f"Test Accuracy: {accuracy:.2f}%")



riemannize(model, 2, exclusions = [model.head]) #Get into shape to use with our optimizer
print(model)

# hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]

# hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]

# nonhidden_params = [*model.head.parameters()]

riemannized_params = [param for name, param in model.named_parameters() if name == "X_riemann"]
unriemannized_params = [param for name, param in model.named_parameters() if name != "X_riemann"]

param_groups = [
    dict(params=riemannized_params, riemann=True,
         lr=0.02, weight_decay=0.01),
    dict(params=unriemannized_params, riemann=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]

optimizer = FrSpecMuon(param_groups)



print("-----------------\n   FrSpecMuon   \n-----------------")
# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(10):

    model.train()

    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        def closure():
            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            return loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss = optimizer.step(closure)
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1} "
        f"Train Loss: {avg_loss:.4f}"
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = 100.0 * correct / total


print("Accuracy after fine tuning:")
print(f"Test Accuracy: {accuracy:.2f}%")







# model = MNISTNet().to(device)


# optimizer = torch.optim.AdamW(model.parameters())

# criterion = nn.CrossEntropyLoss()

# print("-----------------\n   AdamW   \n-----------------")
# # --------------------------------------------------
# # Training loop
# # --------------------------------------------------

# for epoch in range(5):

#     model.train()

#     total_loss = 0.0

#     for x, y in train_loader:
#         x = x.to(device)
#         y = y.to(device)
#         def closure():
#             optimizer.zero_grad()

#             logits = model(x)

#             loss = criterion(logits, y)

#             loss.backward()

#             return loss

#         loss = optimizer.step(closure)

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)

#     print(
#         f"Epoch {epoch+1} "
#         f"Train Loss: {avg_loss:.4f}"
#     )


# # --------------------------------------------------
# # Evaluation
# # --------------------------------------------------

# model.eval()

# correct = 0
# total = 0

# with torch.no_grad():

#     for x, y in test_loader:

#         x = x.to(device)
#         y = y.to(device)

#         logits = model(x)

#         preds = logits.argmax(dim=1)

#         correct += (preds == y).sum().item()
#         total += y.size(0)

# accuracy = 100.0 * correct / total

# print(f"Test Accuracy: {accuracy:.2f}%")