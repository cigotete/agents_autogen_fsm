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

# Sequential Team Operations
# Create an empty directed graph

speaker_transitions_dict = {}
teams = ["A", "B", "C"]
team_size = 5


def get_agent_of_name(agents, name) -> ConversableAgent:
    for agent in agents:
        if agent.name == name:
            return agent


# Create a list of 15 agents 3 teams x 5 agents
agents = [ConversableAgent(name=f"{team}{i}", llm_config=False) for team in teams for i in range(team_size)]

# Loop through each team and add members and their connections
for team in teams:
    for i in range(team_size):
        member = f"{team}{i}"
        # Connect each member to other members of the same team
        speaker_transitions_dict[get_agent_of_name(agents, member)] = [
            get_agent_of_name(agents, name=f"{team}{j}") for j in range(team_size) if j != i
        ]

# Team leaders connection
print(get_agent_of_name(agents, name="B0"))
speaker_transitions_dict[get_agent_of_name(agents, "A0")].append(get_agent_of_name(agents, name="B0"))
speaker_transitions_dict[get_agent_of_name(agents, "B0")].append(get_agent_of_name(agents, name="C0"))

visualize_speaker_transitions_dict(speaker_transitions_dict, agents)