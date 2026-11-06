from matplotlib import pyplot as plt
import os
from PIL import Image
import warnings
from dotenv import load_dotenv

load_dotenv()


# Suppress all warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")
# # # show number or rows
# print(ds.num_rows)

# # # get an image from the dataset
# flower = ds["train"][78]["image"]
# # Display the image using matplotlib
# plt.imshow(flower)
# plt.axis("off")
# plt.show()


def show_image_from_uri(uri):
    # Open the image using PIL
    img = Image.open(uri)

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis("off")  # Turn off axis labels
    plt.show()


# ==== Save all images (500) do directory ====
dataset_folder = "./dataset/flowers-102-categories"
os.makedirs(dataset_folder, exist_ok=True)


# Function to save images
def save_images(dataset, dataset_folder, num_images=500):
    for i in range(num_images):
        print(f"Saving image {i+1} of {num_images}")
        # Get the image data
        image = dataset["train"][i]["image"]

        # Save the image
        image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))

    print(f"Saved the first 500 images to {dataset_folder}")


# save_images(ds, dataset_folder, num_images=500)  # Uncomment to save images


# == Setup the chromaDB ==
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# create a chromadb object
chroma_client = chromadb.PersistentClient(path="./data/flower.db")

# instantiate image loader
image_loader = ImageLoader()

# instantiate multimodal embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# create the collection, - vector database
flower_collection = chroma_client.get_or_create_collection(
    "flowers_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

ids = []
uris = []

### Iterate over each file in the dataset folder -- uncomment this for the first time run ###
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith(".png"):
        file_path = os.path.join(dataset_folder, filename)

        # Append id and uri to respective lists
        ids.append(str(i))
        uris.append(file_path)

# # Assuming multimodal_db is already defined and available
# flower_collection.add(ids=ids, uris=uris)  # Uncomment to add images to the database

# print("Images added to the database.")

# # Validate the VectorDB with .count()
# print(flower_collection.count())  # res: 500


# === Functions for Querying the VectorDB ===
def query_db(query, results=5):
    print(f"Querying the database for: {query}")
    results = flower_collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results


def print_results(results):
    for idx, uri in enumerate(results["uris"][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        # Display the image using matplotlib
        show_image_from_uri(uri)
        print("\n")


#  === For Testing ONlY === uncomment to test
# query = "purple petals"  # Change the query to test different images or different
# results = query_db(query)
# print_results(results)

# =================================
# === Setting up the RAG Flow ===

# 1. the user submits a query (question, query, etc)
# 2. the query is sent to the multimodal database (retrieval function first)
## * in our case, we try to pull the images that match the user's query
# 3. Those images are then passed (along with prompt) to the a vision model where it will use the images context and respond to the prompt as a final output


## MUST install the following libraries:
# pip install langchain-community langchain-core langchain-openai
# =================================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64

# Instantiate the OpenAI model
vision_model = ChatOpenAI(
    model="gpt-4o", temperature=0.0
)  # this model has vision capabilities

# instantiate the output parser
parser = StrOutputParser()


# Define the prompt template
image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented florist and you have been asked to create a bouquet of flowers for a special event. Answer the user's question  using the given image context with direct references to parts of the images provided."
            " Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure.",
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "what are some good ideas for a bouquet arrangement {user_query}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

# Define the LangChain Chain
vision_chain = image_prompt | vision_model | parser


# === Foramtting query results for LLM prompting ===
# to input the images in as context, we need to first encode the images as base64 strings for the LLM to understand


# The function below that will do that, and create a dictionary along with
# the original user query to pass into the chain. The chain will take a dictionary input,
# that will correspond to the three pieces of information
# that need to be injected into it {user_query}, {image_data_1}, {image_data_2}.
def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
    inputs = {}

    # Add user query to the dictionary
    inputs["user_query"] = user_query

    # Get the first two image paths from the 'uris' list
    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    # Encode the first image
    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    # Encode the second image
    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    # inputs dictionary will have the user query and the base64 encoded images and will look like this:
    # {
    #     "user_query": "pink flower with yellow center",
    #     "image_data_1": "base64_encoded_image_1",
    #     "image_data_2": "base64_encoded_image_2"
    # }
    print("Prompt inputs formatted....")
    return inputs


## === Putting it all together ===
print("Welcome to the flower arrangement service!")
print("Please enter your query to get some ideas for a bouquet arrangement.")

query = input("Enter your query: \n")

# Running Retrieval and Generation
results = query_db(query, results=2)
prompt_input = format_prompt_inputs(results, query)
response = vision_chain.invoke(prompt_input)

print("\n ------- \n")

print("\n ---Response---- \n")
print(response)


# Display the retrieved images
print("\n Here are some ideas for a bouquet arrangement based on your query: \n")
show_image_from_uri(results["uris"][0][0])
show_image_from_uri(results["uris"][0][1])

print("\n Images URI: \n")  
print(f"Image 1: {results["uris"][0][0]}")
print(f"Image 2: {results["uris"][0][1]}")
