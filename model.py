import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, device: torch.device, input_size: int, hidden_size: int) -> None:
        super().__init__()

        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size).to(device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = nn.functional.relu(x).to(self.device)
        x = self.fc2(x)
        x = nn.functional.relu(x).to(self.device)
        x = self.fc3(x)
        prob_up = torch.sigmoid(x).to(self.device)
        return prob_up
