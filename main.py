import torch

from game import SnakeGameAI
from agent import Agent

AGENT_1_PATH = "state_dict/VER_2.0.pth"

checkpoint = {
    "state_dict" : None,
    "n_games" : 0,
    "record" : 0
}


def train():
    #load checkpoint
    loaded = None
    try:
        loaded = torch.load(AGENT_1_PATH, weights_only=False)
    except:
        pass
    if loaded is not None:
        checkpoint.update(loaded)
        print("[*] safely loaded")

    record = checkpoint.get("record")
    game = SnakeGameAI()
    agent = Agent(checkpoint.get("state_dict"), checkpoint.get("n_games"))

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score>record:
                record = score

            state_dict = agent.model.state_dict()
            checkpoint.update({"state_dict": state_dict, "n_games": agent.n_games, "record":record})
            torch.save(checkpoint, AGENT_1_PATH)

            print(f"<CP saved> | Game: {agent.n_games} | Score: {score} | Record: {record}")

if __name__ == '__main__':
    train()