"""Train the FaceCNN model and save face_cnn.pth.
Uses train_data.pt (already in the repo) for training.
Runs entirely on CPU - the trained model is tiny (~17M params) and
training over the Olivetti subset (320 imgs, 64x64) takes < 30s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class FaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 16 * 16, 128)
        self.fc2   = nn.Linear(128, 40)

    def forward(self, x):
        x = self.conv1(x); x = F.max_pool2d(x, 2)
        x = self.conv2(x); x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    train_x, train_y = torch.load("train_data.pt")
    print(f"train: x={train_x.shape} y={train_y.shape}")
    # Make sure x is (N,1,64,64) float in [0,1]
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
    train_x = train_x.float()
    if train_x.max() > 1.5:
        train_x = train_x / 255.0
    train_y = train_y.long()

    ds = TensorDataset(train_x, train_y)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model = FaceCNN()
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    EPOCHS = 30
    for ep in range(EPOCHS):
        tot_loss = 0.0; correct = 0; n = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:2d}/{EPOCHS}  loss={tot_loss/n:.4f}  acc={100*correct/n:.1f}%")

    torch.save(model.state_dict(), "face_cnn.pth")
    print("saved face_cnn.pth")

if __name__ == "__main__":
    main()
