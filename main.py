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

# Dictionary defining how are relations between agents.
allowed_speaker_transitions_dict = {
    agents[0]: [agents[1], agents[2], agents[3], agents[4]],
    agents[1]: [agents[0]],
    agents[2]: [agents[0]],
    agents[3]: [agents[0]],
    agents[4]: [agents[0]],
}

# Visualizing relation.
visualize_speaker_transitions_dict(allowed_speaker_transitions_dict, agents)