from openai import OpenAI  # must install openai package
import os

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

model = "gpt-4o-mini"
# == few-shot learning
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a translator."},
        {
            "role": "user",
            "content": """ Translate these sentences: 
            'Hello' -> 'Hola', 
            'Goodbye' -> 'Adiós'. 
            '.
             Now translate: 'Thank you'.""",
        },
    ],
)
# print(completion.choices[0].message.content)

# Direct prompt example with openai / Zero-shot prompting
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)

# print(completion.choices[0].message.content)


# == Chain of thought ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a math tutor."},
        {
            "role": "user",
            "content": "Solve this math problem step by step: If John has 5 apples and gives 2 to Mary, how many does he have left?",
        },
    ],
)
# print(completion.choices[0].message.content)

# == Instructional prompts
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You a knowledgeable personal trainer and writer.",
        },
        {
            "role": "user",
            "content": "Write a 300-word summary of the benefits of exercise, using bullet points.",
        },
    ],
)
# print(completion.choices[0].message.content)

# == Role-playing prompts ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a character in a fantasy novel."},
        {
            "role": "user",
            "content": "Describe the setting of the story.",
        },
    ],
)
# print(completion.choices[0].message.content)

# == Open-ended prompt ==
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a philosopher."},
        {
            "role": "user",
            "content": "What is the meaning of life?",
        },
    ],
)
# print(completion.choices[0].message.content)

# == temperature and top-p sampling
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a creative tagline for a coffee shop."},
    ],
    # temperature=0.9,  # controls the randomness of the output
    top_p=0.9,  # controls the diversity of the output
)
# print(completion.choices[0].message.content)

# === Combining techniques ===
completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a travel blogger."},
        {
            "role": "user",
            "content": "Write a 500-word blog post about your recent trip to Paris. Make sure to give a step-by-step itinerary of your trip.",
        },
    ],
    temperature=0.9,
    stream=True,
    # top_p=0.9,
)
# print(completion.choices[0].message.content)
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content or "", end="")
