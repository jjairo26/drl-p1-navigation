import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from model import QNetwork
from ReplayBuffer import ReplayBuffer
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Agent():
    def __init__(self, state_size, action_size, buffer_size, batch_size, lr, gamma, tau, update_every, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimenstion of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr          # learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau      # for soft update of target parameters
        self.t_step = 0      # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_every = update_every
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.local_network_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state, eps=0.01):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state) # Q-values (tensor) for the current state for all actions
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) #data.numpy needs the action values to be on CPU
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # learn from sampled subset if there are enough samples in memory
            if len(self.memory) > self.memory.batch_size:
                # Sample random minibatch of transitions from D
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples."""

        states, actions, rewards, next_states, dones = experiences

        # -- Calculate target y = r_j + gamma * max ^Q (s', a; theta^-) (only r_j if done) --
        # Get max predicted Q values for next states from target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # actions is a column tensor [batch_size, 1], 
        # gather will select the Q-values corresponding to the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions) 

        # Compute loss (MSE loss between expected and target Q values)
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.local_network_optimizer.zero_grad()
        loss.backward() # Compute gradients
        self.local_network_optimizer.step() # Update local model parameters

        # Update target network (this is done instead of a hard update every C steps)
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
