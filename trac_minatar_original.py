#Update MinAtar
'%pip install minatar'

import minatar.environment

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from src.utils.device_manager import get_device_manager, get_device_info
from typing import Tuple, Any, Callable, Dict
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# GPU SUPPORT: Initialize device manager early
try:
    DEVICE_MANAGER = get_device_manager()
    DEVICE_INFO = get_device_info()
    print(f"🔧 GPU Support: {DEVICE_INFO['device']}")
    if DEVICE_INFO['is_cuda']:
        print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
        print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
except Exception as e:
    print(f"GPU Support: Failed to initialize device manager: {e}")
    DEVICE_MANAGER = None
    DEVICE_INFO = {'device': 'cpu', 'is_cuda': False, 'is_gpu_available': False}

# Set CUDA device based on arguments
# Initialize device variable at global scope
device = None
if DEVICE_INFO['is_cuda'] and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {device}")
else:
    device = torch.device("cpu")
    print(f"CUDA is not available or disabled. Using CPU: {device}")

"""
Control Tasks
"""
# opt will be set from command line arguments

# num distribution shifts
levels = 10

# Hyperparameters
lr = 0.01
max_episodes = 2
train_epochs = 40
max_timesteps = 1000
state_scale = 1.0
reward_scale = 20.0
batch_size = 32
gamma = 0.95
lambd = 0.95

# when to introduce distribution shift
level_switch = 200
max_iterations = levels * level_switch

# peturbation range
rp_range = 2

# SET THE SEED
seed = 0
random.seed(seed)
np.random.seed(seed)

"""
Base PPO Setup
"""
class PolicyNetwork(torch.nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, n)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        y = self.fc4(x)
        y = F.softmax(y, dim=-1)
        return y

    def sample_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item()

    def best_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state).squeeze()
        action = torch.argmax(y)

        return action.item()

    def evaluate_actions(self, states, actions):
        y = self(states)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy
class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 1)
        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)

        return y.squeeze(1)

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()
def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]
class Episode:
    def __init__(self, gamma=gamma, lambd=lambd):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, observation, action, reward, value, log_probability, reward_scale=reward_scale
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)
        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]
def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()
class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
def ac_loss_clipped(new_log_probabilities, old_log_probabilities, advantages, epsilon_clip=0.2):
    probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_probabiliy_ratios = torch.clamp(
        probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip
    )

    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probabiliy_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)
def train_combined_networks(policy_model, value_model, combined_optimizer, data_loader, epochs=train_epochs, clip=0.2):
    c1 = 0.01  # Coefficient for entropy regularization
    c2 = 0.5   # Coefficient for value loss weight

    for epoch in range(epochs):
        policy_losses = []
        value_losses = []

        for observations, actions, advantages, log_probabilities, rewards_to_go in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            combined_optimizer.zero_grad()

            new_log_probabilities, entropy = policy_model.evaluate_actions(observations, actions)
            policy_loss = (
                ac_loss_clipped(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=clip,
                ).mean()
                - c1 * entropy.mean()
            )
            policy_losses.append(policy_loss.item())

            values = value_model(observations)
            value_loss = c2 * F.mse_loss(values, rewards_to_go)
            value_losses.append(value_loss.item())

            total_loss = policy_loss + value_loss

            total_loss.backward()
            combined_optimizer.step()

"""
TRAC Optimizer - Erfi function
"""
def polyval(x,coeffs):
    """Implementation of the Horner scheme to evaluate a polynomial

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124

    Args:
        x (torch.Tensor): variable
        coeffs (torch.Tensor): coefficients of the polynomial
    """
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x[0]
    return(curVal+coeffs[len(coeffs)-1])
# Complex number error computation - forward and backward pass
class ERF_1994(torch.nn.Module):
    """Class to compute the error function of a complex number (extends torch.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """
    def __init__(self, n_coefs):
        """Defaul constructor

        Args:
            n_coefs (integer): The number of polynomial coefficients to use in the approximation
        """
        super(ERF_1994, self).__init__()
        # compute polynomial coefficients and other constants
        self.N = n_coefs
        self.i = torch.complex(torch.tensor(0.),torch.tensor(1.))
        self.M = 2*self.N
        self.M2 = 2*self.M
        self.k = torch.linspace(-self.M+1, self.M-1, self.M2-1)
        self.L = torch.sqrt(self.N/torch.sqrt(torch.tensor(2.)))
        self.theta = self.k*torch.pi/self.M
        self.t = self.L*torch.tan(self.theta/2)
        self.f = torch.exp(-self.t**2)*(self.L**2 + self.t**2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real/self.M2
        self.a = torch.flipud(self.a[1:self.N+1])

    def w_algorithm(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L+self.i*z)/(self.L-self.i*z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2*p/(self.L-self.i*z)**2+(1/torch.sqrt(torch.tensor(torch.pi)))/(self.L-self.i*z)
        return w

    def forward(self, z):
        """Compute the error function of a complex number

        The result is computed by manipulating the Faddeeva function.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: erf(z) for each element of z
        """
        # exploit the symmetry of the error function
        # find the sign of the real part
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        # flip sign of imaginary part if negative
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z*self.i)) - z**2) + 1
        return torch.complex(out.real*sign_r, out.imag*sign_i)

    def backward(self, z):
        """Compute the gradient of the error function of a complex number.

        As we know the analytical derivative of the the error function, we can use it directly.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)
        Returns:
            torch.Tensor: grad(erf(z)) for each element of x
        """
        return 2/torch.sqrt(torch.tensor(torch.pi))*torch.exp(-z**2)
erf_torch = ERF_1994(128)
def erfi(x):
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)

    # Convert x to a complex tensor where the real part is zero
    ix = torch.complex(torch.zeros_like(x), x)

    # Compute erf(ix) / i
    erfi_x = erf_torch(ix).imag  # Extract the imaginary part of erf(ix)
    return erfi_x

"""
TRAC Wrapper
"""
def _init_state(
        optimizer: torch.optim.Optimizer,
        theta_ref: Dict[torch.Tensor, torch.Tensor],
        betas: Tuple[float],
        s_prev: float,
        eps: float):
    if '_trac' not in optimizer.state:
        optimizer.state['_trac'] = {
            'betas': torch.tensor(betas),
            's_prev': torch.tensor(s_prev),
            'eps': eps,
            's': torch.zeros(len(betas)),
            'theta_ref': {},
            'variance': torch.zeros(len(betas)),
            'sigma': torch.full((len(betas),), 1e-8),
            'iter_count': 0,
        }
        _init_reference(optimizer, theta_ref)
def _init_reference(
        optimizer: torch.optim.Optimizer,
        theta_ref: Dict[torch.Tensor, torch.Tensor],):
    '''
    Args:
        optimizer: optimizer instance to store reference for.
        theta_ref: mapping of parameters to their initial values at the start of optimization.
    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state['_trac'][p] = {
                'ref': theta_ref[p].clone(),
            }         
def _step(
        optimizer: torch.optim.Optimizer,
        base_step: Callable,
        betas: Tuple[float],
        s_prev: float,
        eps: float,
        ):
    '''
    Args:
        optimizer: trac optimizer instance
        base_step: The "step" function of the base optimizer
        betas: list of beta values.
        s_init: initial scale value.
        eps: epsilon value.
    '''

    prev_grad = torch.is_grad_enabled()


    torch.set_grad_enabled(False)
    updates = {}
    grads = {}
    deltas = {}

    for group in optimizer.param_groups:
        for p in group['params']:

            if p.grad is None:
                grads[p] = None
            else:
                grads[p] = p.grad.clone()
            updates[p] = p.data.clone()

    torch.set_grad_enabled(prev_grad)
    result = base_step(None)
    torch.set_grad_enabled(False)
    
    _init_state(optimizer, updates, betas, s_prev, eps)
    trac_state = optimizer.state['_trac']


    for group in optimizer.param_groups:
        for p in group['params']:
            if grads[p] is None:
                continue

            theta_ref = trac_state[p]['ref']

            deltas[p] = (updates[p] - theta_ref)/(torch.sum(trac_state['s']) + trac_state['eps'])

            updates[p].copy_(p-updates[p])

    h = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            grad = grads[p]

            delta = deltas[p]
            product = torch.dot(delta.flatten(), grad.flatten())
            if product.isnan():
                raise ValueError("NaNs in product")
            h += product

            delta.add_(updates[p])

    device = h.device

    for key in trac_state:
        try:
            if trac_state[key].device != device:
                trac_state[key] = trac_state[key].to(device)
        except:
            pass

    s = trac_state['s']
    s_prev = trac_state['s_prev']
    betas = trac_state['betas']
    eps = trac_state['eps']
    variance = trac_state['variance'] 
    sigma = trac_state['sigma']                                 
    trac_state['iter_count'] += 1

    variance.mul_(
        betas**2).add_(torch.square(h))
    sigma.mul_(betas).sub_(h)
    f_term = s_prev / (erfi(torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))))
    s_term = erfi(sigma / (torch.sqrt(torch.tensor(2.0)) * torch.sqrt(variance) + eps))
    if (f_term * s_term).isnan().any():
        raise ValueError("NaNs in s")
    s.copy_(f_term * s_term)

    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            theta_ref = trac_state[p]['ref']
            delta = deltas[p]
            s_sum = torch.sum(s)

            scale = max(s_sum, 0.0)
            p.copy_(theta_ref + delta * scale)

    log_data = {
        'iter_count': trac_state['iter_count'],
        's': torch.sum(s).item(),
    }

    torch.set_grad_enabled(prev_grad)
    return result, log_data
class trac:
    pass
def is_trac(opt):
    return isinstance(opt, trac)
# Wraps the base opt with trac
def start_trac(
        log_file,
        Base: Any,
        betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999,
                               0.99999, 0.999999),
        s_prev: float = 1e-8,
        eps: float = 1e-8,
        ):

    class TRACOPT(Base, trac):
        '''
        Wraps the base opt with trac.
        
        '''

        def step(self):
            result, log_data = _step(self, super().step, betas, s_prev, eps)
            with open (log_file, 'a') as f:
                f.write(str(log_data) + '\n')
            return result

    TRACOPT.__name__ += Base.__name__

    return TRACOPT


"""
Training functions
"""
def get_perturbations(env_name, seed, levels, rp_range = rp_range):    
  env = minatar.environment.Environment(env_name)
  observation = env.state().flatten().astype(np.float32)
  random_perturbations = [
        np.random.normal(0, rp_range, observation.shape) for _ in range(levels)
    ]
  # make the first random perturbation zero
  random_perturbations[0] = np.zeros(observation.shape)
  return random_perturbations

def get_reward(env, env_name, prev_state, current_state):
    """Calculate reward based on MinAtar game state changes"""
    if env_name == "breakout":
        # Reward +1 for each brick broken (detected by brick count decrease)
        prev_bricks = np.sum(prev_state[:, :, 3])  # Brick channel
        curr_bricks = np.sum(current_state[:, :, 3])
        return 1.0 if curr_bricks < prev_bricks else 0.0
    
    elif env_name == "asterix":
        # Reward +1 for picking up treasure (detected by gold count decrease)
        prev_gold = np.sum(prev_state[:, :, 3])  # Gold channel
        curr_gold = np.sum(current_state[:, :, 3])
        return 1.0 if curr_gold < prev_gold else 0.0
    
    elif env_name == "space_invaders":
        # Reward +1 for shooting aliens (detected by alien count decrease)
        prev_aliens = np.sum(prev_state[:, :, 1])  # Alien channel
        curr_aliens = np.sum(current_state[:, :, 1])
        return 1.0 if curr_aliens < prev_aliens else 0.0
    
    else:
        return 0.0
def train(env_name, opt_choice, random_perturbations, levels, seed, no_noise = False):
    global device  # Access the globally defined device variable

    # Create log txt files
    suffix = "_no_noise" if no_noise else ""
    trac_reward_log_file = f'iclr/original/minatar/trac_reward_log_{env_name}_{seed}{suffix}.txt'
    base_reward_log_file = f'iclr/original/minatar/base_reward_log_{env_name}_{seed}{suffix}.txt'

    # Setup env
    env = minatar.environment.Environment(env_name)
    observation = env.state().flatten().astype(np.float32)
    n_actions = env.num_actions()
    feature_dim = observation.shape[0]
    max_iterations = levels * level_switch

    tqdm_bar = tqdm(range(max_iterations), desc="Training", unit="iteration")

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)

    suffix = "_no_noise" if no_noise else ""
    if opt_choice == "TRAC":
        trac_combined_optimizer = start_trac(log_file=f'iclr/original/minatar/trac_{env_name}_{seed}{suffix}.text', Base=optim.Adam)(
            [
                {"params": policy_model.parameters(), "lr": lr},
                {"params": value_model.parameters(), "lr": lr},
            ]
        )
        combined_optimizer = trac_combined_optimizer
        reward_log_file = trac_reward_log_file
        print(f"USING TRAC. Log file: iclr/original/minatar/trac_{env_name}_{seed}{suffix}.text")
    else:  # base optimizer
        # For base optimizer, we don't need the TRAC log file
        trac_combined_optimizer = None
        base_combined_optimizer = torch.optim.Adam(
            [
                {"params": policy_model.parameters(), "lr": lr},
                {"params": value_model.parameters(), "lr": lr},
            ]
        )
        combined_optimizer = base_combined_optimizer
        reward_log_file = base_reward_log_file
        print(f"USING BASE ADAM. Log file: iclr/original/minatar/base_reward_log_{env_name}_{seed}{suffix}.txt")
    history = History()
    level = 0
    for ite in tqdm_bar:
        # Switch perturbation level
        if ite % level_switch == 0:
            random_perturbation = random_perturbations[level]
            level += 1

        episodes_reward = []

        for _ in range(max_episodes):
            observation = env.state().flatten().astype(np.float32)
            observation += random_perturbation
            episode = Episode()

            for timestep in range(max_timesteps):
                action, log_probability = policy_model.sample_action(observation / state_scale)
                value = value_model.state_value(observation / state_scale)

                # Store previous state for reward calculation
                prev_state = env.state()
                env.act(action)
                current_state = env.state()
                reward = get_reward(env, env_name, prev_state, current_state)
                done = env.env.terminal
                new_observation = current_state.flatten().astype(np.float32)
                new_observation += random_perturbation

                episode.append(
                    observation=observation / state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale,
                )

                observation = new_observation

                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1:
                    value = value_model.state_value(observation / state_scale)
                    episode.end_episode(last_value=value)

            episodes_reward.append(reward_scale * np.sum(episode.rewards))
            history.add_episode(episode)

        mean_rewards = np.mean(episodes_reward)
        tqdm_bar.set_postfix(mean_rewards=mean_rewards)

        with open(reward_log_file, 'a') as f:
            f.write(str(mean_rewards) + '\n')
        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)
        train_combined_networks(policy_model, value_model, combined_optimizer, data_loader, train_epochs)
        history.free_memory()


# Test to get quick one seed results for MinAtar environments
def test_breakout(levels, seed, no_noise = False):
    peturbations = get_perturbations("breakout", seed, levels, rp_range = 0 if no_noise else rp_range)
    print("Online Peturbations are")
    print(peturbations)
    env = "breakout"
    train(env, opt, peturbations, levels, seed, no_noise)
    plot(env, seed, no_noise, opt)

def test_asterix(levels, seed, no_noise = False):
    peturbations = get_perturbations("asterix", seed, levels, rp_range = 0 if no_noise else rp_range)
    print("Online Peturbations are")
    print(peturbations)
    env = "asterix"
    train(env, opt, peturbations, levels, seed, no_noise)
    plot(env, seed, no_noise, opt)

def test_space_invaders(levels, seed, no_noise = False):
    peturbations = get_perturbations("space_invaders", seed, levels, rp_range = 0 if no_noise else rp_range)
    print("Online Peturbations are")
    print(peturbations)
    env = "space_invaders"
    train(env, opt, peturbations, levels, seed, no_noise)
    plot(env, seed, no_noise, opt)


"""
Plot the results
"""
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data
def plot(env_name, seed, no_noise = False, opt_choice = "base"):
    # Read data from files based on optimizer choice
    suffix = "_no_noise" if no_noise else ""
    if opt_choice == "TRAC":
        data_file = f'iclr/original/minatar/trac_reward_log_{env_name}_{seed}{suffix}.txt'
        label = 'TRAC PPO'
        color = '#b71540'
        plot_file = f'iclr/original/minatar/trac_reward_plot_{env_name}_{seed}{suffix}.png'
    else:  # base optimizer
        data_file = f'iclr/original/minatar/base_reward_log_{env_name}_{seed}{suffix}.txt'
        label = 'Adam PPO'
        color = '#4a69bd'
        plot_file = f'iclr/original/minatar/base_reward_plot_{env_name}_{seed}{suffix}.png'

    # Read and convert data to float
    try:
        data = read_data(data_file)
        data = [float(i) for i in data]
    except FileNotFoundError:
        print(f"Warning: Could not find log file {data_file}")
        return

    # Smooth data
    window = 5
    data = np.convolve(data, np.ones(window) / window, mode='valid')

    # Create a plot with seaborn
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))

    plt.plot(data, label=label, color=color)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title(f'{env_name}', fontsize=24)
    plt.legend()
    plt.show()

    # save plots
    plt.savefig(plot_file)


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRAC Original Training')
    parser.add_argument('--task', type=str, choices=['breakout', 'asterix', 'space_invaders'], 
                       default='breakout', help='Choose task: breakout, asterix, or space_invaders')
    parser.add_argument('--levels', type=int, default=2, help='Number of training levels')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--no_noise', action='store_true', help='Run without noise (ablation study)')
    parser.add_argument('--optimizer', type=str, choices=['TRAC', 'base'], 
                       default='base', help='Choose optimizer: TRAC or base (Adam)')
    
    args = parser.parse_args()
    
    levels = args.levels
    seed = args.seed
    no_noise = args.no_noise
    opt = args.optimizer  # Set the global opt variable
    
    # Select task based on command line argument
    if args.task == 'breakout':
        test_breakout(levels, seed, no_noise)
        print(f"Tested Breakout with {levels} levels, seed {seed}, optimizer {opt}" + (" (no noise)" if no_noise else ""))
    elif args.task == 'asterix':
        test_asterix(levels, seed, no_noise)
        print(f"Tested Asterix with {levels} levels, seed {seed}, optimizer {opt}" + (" (no noise)" if no_noise else ""))
    elif args.task == 'space_invaders':
        test_space_invaders(levels, seed, no_noise)
        print(f"Tested Space Invaders with {levels} levels, seed {seed}, optimizer {opt}" + (" (no noise)" if no_noise else ""))


