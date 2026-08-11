import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PatchFeatureSVDDNet(nn.Module):
    def __init__(self, input_dim, rep_dim):
        super().__init__()
        self.rep_dim = rep_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LeakyReLU(),
            nn.Linear(16, rep_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def init_center_c(net, train_loader, device, eps=0.1):
    c = torch.zeros(net.rep_dim, device=device)
    n_samples = 0

    net.eval()

    for (x,) in train_loader:
        x = x.to(device)
        outputs = net(x)

        n_samples += outputs.shape[0]
        c += outputs.sum(dim=0)

    c /= n_samples

    c[(torch.abs(c) < eps) & (c < 0)] = -eps
    c[(torch.abs(c) < eps) & (c > 0)] = eps

    return c


def fit_deep_svdd(
    train_features,
    output_dim,
    device,
    epochs=150,
    lr=1e-4,
    batch_size=256,
    weight_decay=5e-7,
):
    train_tensor = torch.as_tensor(
        train_features,
        dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    net = PatchFeatureSVDDNet(
        input_dim=train_features.shape[1],
        rep_dim=output_dim,
    ).to(device)

    c = init_center_c(net, loader, device)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        count = 0

        for (x,) in loader:
            x = x.to(device)

            optimizer.zero_grad()

            z = net(x)
            dist = torch.sum((z - c) ** 2, dim=1)

            loss = torch.mean(dist)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            count += len(x)

        print(
            f"[DeepSVDD] epoch {epoch+1:03d}/{epochs} "
            f"loss={total_loss/count:.8f}"
        )

    return net, c


@torch.no_grad()
def transform_deep_svdd(net, features, device, batch_size=4096):
    tensor = torch.as_tensor(features, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    outputs = []

    net.eval()

    for (x,) in loader:
        outputs.append(
            net(x.to(device)).cpu()
        )

    return torch.cat(outputs, dim=0).numpy()
