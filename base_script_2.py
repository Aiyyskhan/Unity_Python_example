import numpy as np
import mlagents
from mlagents_envs.environment import UnityEnvironment as UE

class PseudoBrain:
	def __init__(self, n_outputs):
		self.n_outputs = n_outputs

	def __call__(self, inputs):
		_ = inputs
		return np.random.uniform(
			low=-1.0, high=1.0, size=self.n_outputs
		).astype(np.float32)

env = UE(file_name='Ex_1', seed=1, side_channels=[])

env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

# print("Форма наблюдений : ", len(spec.observation_shapes))
print("Форма наблюдений : ", spec.observation_shapes[0][0])
print("Форма действий : ", spec.action_shape)
print("Тип действий : ", spec.action_type)

if spec.is_action_continuous():
	print("The action is continuous")

if spec.is_action_discrete():
	print("The action is discrete")

decision_steps, terminal_steps = env.get_steps(behavior_name)
print(f"Кол-во агентов: {len(decision_steps)}")
print(f"Порт редактора по умолчанию: {env.DEFAULT_EDITOR_PORT}")
print(f"Агент ID's: {decision_steps.agent_id}")
# print(decision_steps.obs)

action_shape = spec.action_shape

# brains
brains = {decision_steps.agent_id[idx] : PseudoBrain(spec.action_shape) for idx in range(len(decision_steps))}

episode = 0
episode_rewards = 0 # For the tracked_agent
done = False

while episode < 150:
	# env.reset()
	decision_steps, terminal_steps = env.get_steps(behavior_name)
	episode_rewards += np.sum(decision_steps.reward)
	if episode % 10 == 0:
		print(f"Total rewards for episode {episode} is {episode_rewards}")
		# print(f"Кол-во агентов: {len(decision_steps)}")
		# print(f"Кол-во агентов term: {len(terminal_steps)}")
		# print(f"Action mask: {decision_steps.action_mask}")
		# print(f"Interrupted: {terminal_steps.interrupted}")

	# Generate an action for all agents
	# action = spec.create_random_action(len(decision_steps))

	actions = np.zeros((len(decision_steps), action_shape))
	for idx, agent_id in enumerate(decision_steps):
		actions[idx] = brains[agent_id](decision_steps[agent_id].obs[0])

	# Set the actions
	env.set_actions(behavior_name, actions)
	# Move the simulation forward
	env.step()

	episode += 1

env.close()
print("Closed environment")
