import json
import os
from openai import OpenAI

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 1. Define your function (your schema)
functions = [
    {
        "name": "extract_user_info",
        "description": "Get user information from text",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "company": {"type": "string"}
            }
        }
    }
]

# 2. Call the model with the function definition
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "John Doe is the CEO of Acme Inc. Email: john.doe@acme.com"}],
    tools=[
        {
            "type": "function",
            "function": functions[0]
        }
    ],
    tool_choice="auto"
)

# 3. The model returns a structured JSON object to call your function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = tool_call.function.arguments
    user_data = json.loads(arguments)
    print(json.dumps(user_data, indent=2))
    # Output will be:
    # '{"name": "John Doe", "email": "john.doe@acme.com", "company": "Acme Inc."}'
else:
    print("No tool call in response")
