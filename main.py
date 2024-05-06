import random
import autogen
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.graph_utils import visualize_speaker_transitions_dict

config_list_gpt3 = autogen.config_list_from_json(
  "OAI_CONFIG_LIST.json",
  filter_dict={
    "model": ["gpt-3.5-turbo-1106"]
  }
)

llm_config_gpt3 = {
  "temperature": 0,
  "timeout": 300,
  "seed": random.randint(100, 100000),
  "config_list": config_list_gpt3
}

# List of agents
agents = [ConversableAgent(name=f"Agent{i}", llm_config=False) for i in range(5)]

allowed_speaker_transitions_dict = {}

# Filling dictionary with each agent as a key
for agent in agents:
    transitions = []

    # Add each agent to the list of possible destinations
    for other_agent in agents:
        transitions.append(other_agent)

    # Assign the list of transitions as the value in the dictionary for the current 'agent' key
    allowed_speaker_transitions_dict[agent] = transitions

# Result: Dictionary where each agent can transfer to any other agent

# Printing agents relations.
visualize_speaker_transitions_dict(allowed_speaker_transitions_dict, agents)