import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output_1 = F.relu(self.layer1(x))
        output_2 = F.relu(self.layer2(output_1))
        output = self.layer3(output_2)
        return output


class QTrainer:
    def __init__(self, model, gamma, l_rate):
        self.model = model
        self.gamma = gamma
        self.lr = l_rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.as_tensor(state, dtype=torch.float)
        action = torch.as_tensor(action, dtype=torch.float)
        reward = torch.as_tensor(reward, dtype=torch.float)
        next_state = torch.as_tensor(next_state, dtype=torch.float)

        game_over = torch.as_tensor(game_over, dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = torch.unsqueeze(game_over, 0)

        pred = self.model(state)
        target = pred.clone()

        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_pred = torch.max(next_pred, dim=1)[0]

        Q_new = reward + self.gamma * max_next_pred * (~game_over)

        action_indices = torch.argmax(action, dim=1)
        batch_indices = torch.arange(len(state))
        target[batch_indices, action_indices] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()