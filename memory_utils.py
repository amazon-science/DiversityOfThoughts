import os
import json
import boto3
import openai
import logging
import jsonlines
import numpy as np
from tqdm import tqdm
from typing import List
from termcolor import colored

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = openai.OpenAI()


def generate_text_embeddings(model_id, body):
    """
    Generate text embedding by using the Cohere Embed model.
    Args:
        model_id (str): The model ID to use.
        body (str) : The reqest body to use.
    Returns:
        dict: The response from the model.
    """

    logger.info(
        "Generating text emdeddings with the Cohere Embed model %s", model_id)

    accept = '*/*'
    content_type = 'application/json'

    bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-west-2") 

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept=accept,
        contentType=content_type
    )

    logger.info("Successfully generated text with Cohere model %s", model_id)

    return response


def get_cohere_embedding(reflections: list, verbose=False):
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    model_id = 'cohere.embed-english-v3'
    input_type = "clustering" 
    embedding_types = ["float"] 
    embeddings = None

    try:

        body = json.dumps({
            "texts": reflections,
            "input_type": input_type,
            "embedding_types": embedding_types}
        )
        response = generate_text_embeddings(model_id=model_id,
                                            body=body)

        response_body = json.loads(response.get('body').read())

        if verbose:
            print(f"ID: {response_body.get('id')}")
            print(f"Response type: {response_body.get('response_type')}")

        embeddings = np.array(response_body['embeddings'][embedding_types[0]])

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        if verbose:
            print("A client error occured: "+ format(message))
    else:
        if verbose:
            print(f"Finished generating text embeddings with Cohere model {model_id}.")
            
    return embeddings


def generate_openai_text_embeddings(model_id, body):
    """
    Generate text embedding using OpenAI's Embedding model.
    Args:
        model_id (str): The model ID to use (e.g., 'text-embedding-ada-002').
        body (str): The request body to use.
    Returns:
        dict: The response from OpenAI's API.
    """
    logger.info("Generating text embeddings with OpenAI model %s", model_id)

    try:
        response = client.embeddings.create(
            model=model_id,
            input=body
        )
        logger.info("Successfully generated embeddings with OpenAI model %s", model_id)
        return response

    except Exception as e:
        logger.error("Error generating embeddings: %s", str(e))
        return None


def get_openai_embedding(reflections: list, verbose=False):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_id = 'text-embedding-3-small'
    embeddings = None

    try:
        # Call OpenAI's API to get embeddings
        response = generate_openai_text_embeddings(model_id=model_id, body=reflections)

        if response:
            embeddings = np.array(response.data[0].embedding)

            if verbose:
                print(f"Finished generating text embeddings with OpenAI model {model_id}.")
                print(f"Embeddings: {embeddings}")

    except Exception as err:
        logger.error("An error occurred: %s", str(err))
        if verbose:
            print(f"An error occurred: {str(err)}")

    return embeddings


def get_top_k_closest(trajectories, query_embedding, k=1, similarity_axis = "prompt_embedding"):
    """get the top k closest embeddings to the query embedding in cosine space
    """
    
    if similarity_axis == "refection_embedding":
        trajectories = [traj for traj in trajectories if "refection_embedding" in traj.keys()]
    
    embeddings = np.array([i[similarity_axis][:, None] for i in trajectories]).squeeze()
    cosine_similarities = np.dot(embeddings, query_embedding).squeeze()
    top_k_indices = np.argsort(-cosine_similarities)[:k]
    return top_k_indices, cosine_similarities


def get_random_k_indices(trajectories, k=1, similarity_axis="prompt_embedding"):
    """get k random indices from the trajectories
    """
    
    if similarity_axis == "refection_embedding":
        trajectories = [traj for traj in trajectories if "refection_embedding" in traj.keys()]
    
    total_trajectories = len(trajectories)
    random_indices = np.random.choice(total_trajectories, size=k, replace=False)
    
    return random_indices, None