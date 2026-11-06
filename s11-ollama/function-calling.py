import os
import json
import asyncio
import random
from ollama import AsyncClient


# Load the grocery list from a text file
def load_grocery_list(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []
    with open(file_path, "r") as file:
        items = [line.strip() for line in file if line.strip()]
    return items


# Function to fetch price and nutrition data for an item
async def fetch_price_and_nutrition(item):
    print(f"Fetching price and nutrition data for '{item}'...")
    # Replace with actual API calls
    # For demonstration, we'll return mock data
    await asyncio.sleep(0.1)  # Simulate network delay
    return {
        "item": item,
        "price": f"${random.uniform(1, 10):.2f}",
        "calories": f"{random.randint(50, 500)} kcal",
        "fat": f"{random.randint(1, 20)} g",
        "protein": f"{random.randint(1, 30)} g",
    }


# Function to fetch a recipe based on a category
async def fetch_recipe(category):
    print(f"Fetching a recipe for the '{category}' category...")
    # Replace with actual API calls to a recipe site
    # For demonstration, we'll return mock data
    await asyncio.sleep(0.1)  # Simulate network delay
    return {
        "category": category,
        "recipe": f"Delicious {category} dish",
        "ingredients": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
        "instructions": "Mix ingredients and cook.",
    }


async def main():
    # Load grocery list
    grocery_items = load_grocery_list("./data/grocery_list.txt")
    if not grocery_items:
        print("Grocery list is empty or file not found.")
        return

    # Initialize Ollama client
    client = AsyncClient()

    # Define the functions (tools) for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_price_and_nutrition",
                "description": "Fetch price and nutrition data for a grocery item",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "string",
                            "description": "The name of the grocery item",
                        },
                    },
                    "required": ["item"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_recipe",
                "description": "Fetch a recipe based on a category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The category of food (e.g., Produce, Dairy)",
                        },
                    },
                    "required": ["category"],
                },
            },
        },
    ]

    # Step 1: Categorize items using the model
    categorize_prompt = f"""
You are an assistant that categorizes grocery items.

**Instructions:**

- Return the result **only** as a valid JSON object.
- Do **not** include any explanations, greetings, or additional text.
- Use double quotes (`"`) for all strings.
- Ensure the JSON is properly formatted.
- The JSON should have categories as keys and lists of items as values.

**Example Format:**

{{
  "Produce": ["Apples", "Bananas"],
  "Dairy": ["Milk", "Cheese"]
}}

**Grocery Items:**

{', '.join(grocery_items)}
"""

    messages = [{"role": "user", "content": categorize_prompt}]
    # First API call: Categorize items
    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,  # No function calling needed here, but included for consistency
    )

    # Add the model's response to the conversation history
    messages.append(response["message"])
    print(response["message"]["content"])

    # Parse the model's response
    assistant_message = response["message"]["content"]

    try:
        categorized_items = json.loads(assistant_message)
        print("Categorized items:")
        print(categorized_items)

    except json.JSONDecodeError:
        print("Failed to parse the model's response as JSON.")
        print("Model's response:")
        print(assistant_message)
        return

    # Step 2: Fetch price and nutrition data using function calling

    # Construct a message to instruct the model to fetch data for each item
    # We'll ask the model to decide which items to fetch data for by using function calling
    fetch_prompt = """
    For each item in the grocery list, use the 'fetch_price_and_nutrition' function to get its price and nutrition data.
    """

    messages.append({"role": "user", "content": fetch_prompt})

    # Second API call: The model should decide to call the function for each item
    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )
    # Add the model's response to the conversation history
    messages.append(response["message"])

    # Process function calls made by the model
    if response["message"].get("tool_calls"):
        print("Function calls made by the model:")
        available_functions = {
            "fetch_price_and_nutrition": fetch_price_and_nutrition,
        }
        # Store the details for later use
        item_details = []
        for tool_call in response["message"]["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            function_to_call = available_functions.get(function_name)
            if function_to_call:
                result = await function_to_call(**arguments)
                # Add function response to the conversation
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )
                item_details.append(result)

                print(item_details)
    else:
        print(
            "The model didn't make any function calls for fetching price and nutrition data."
        )
        return

    # Step 3: Fetch a recipe for a random category using function calling

    # Choose a random category
    random_category = random.choice(list(categorized_items.keys()))
    recipe_prompt = f"""
    Fetch a recipe for the '{random_category}' category using the 'fetch_recipe' function.
    """
    messages.append({"role": "user", "content": recipe_prompt})

    # Third API call: The model should decide to call the 'fetch_recipe' function
    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )

    # Add the model's response to the conversation history
    messages.append(response["message"])
    # Process function calls made by the model
    if response["message"].get("tool_calls"):
        available_functions = {
            "fetch_recipe": fetch_recipe,
        }
        for tool_call in response["message"]["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            function_to_call = available_functions.get(function_name)
            if function_to_call:
                result = await function_to_call(**arguments)
                # Add function response to the conversation
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )
    else:
        print("The model didn't make any function calls for fetching a recipe.")
        return

    # Final API call: Get the assistant's final response
    final_response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )

    print("\nAssistant's Final Response:")
    print(final_response["message"]["content"])


# Run the async main function
asyncio.run(main())
