import os

from mistralai import Mistral

client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

response = client.beta.conversations.start(
    agent_id="ag_019b4f89eb4b70a8ae4f2d07feffcb89",
    inputs="Hello there!",
)

print(response)