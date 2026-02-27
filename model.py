import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch


#Linear
class QNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)

        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output_1 = F.relu(self.layer1(x))
        output = self.layer2(output_1)
        return output

class QTrainer:
    def __init__(self, model, gamma, l_rate):
        self.model = model
        self.gamma = gamma
        self.l_rate = l_rate
        self.optimizer = optimizer.Adam(model.parameters(), lr=self.l_rate)
        self.cost = nn.MSELoss()

    # Note: everything(well... almost) is connected! pred <-(line:154)-> model <-(line:171)-> loss
    #       and everything is a list since they are batches
    def train_step(self, state, action, reward, next_state, game_over):

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        game_over = torch.tensor(game_over, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        pred = self.model(state)
        target = pred.clone()

        #Each idx is a move from a batch of N random moves from the Memory bank on which we will train.
        for idx in range(len(game_over)):
            Q_new = reward[idx]

            if not game_over[idx]:

                best_outcome = torch.max(self.model(next_state))
                Q_new = reward[idx] + self.gamma*best_outcome

            action_taken_i = torch.argmax(action[idx]).item()
            target[idx][action_taken_i] = Q_new #

        self.optimizer.zero_grad()

        loss = self.cost(target, pred) #take the cost in account
        loss.backward()                #calculate the direction to the minimum (like GD)
        self.optimizer.step()          #backpropogation. take the step down the hill

