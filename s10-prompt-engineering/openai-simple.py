from openai import OpenAI  # must install openai package
import os

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # this is another way to initialize the client

response = client.chat.completions.create(
    model="gpt-6-luna",  # gpt-4o-mini shuts down Jan 14, 2027
    reasoning_effort="none",  # fast answers, like gpt-4o-mini
    messages=[
        {"role": "system", "content": "You are a eastern poet."},
        {
            "role": "user",
            "content": """write me a short poem about the moon. 
               Write the poem in the style of a haiku.
               Make sure include a title for the poem.""",
        },
    ],
)

print(response.choices[0].message.content)
