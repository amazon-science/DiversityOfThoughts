from typing import List, Union, Optional, Literal
import dataclasses

import sys
sys.path.append('../')
import gpt_usage
from termcolor import colored

#claude specific imports
import boto3
from botocore.exceptions import ClientError

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI

import re

def remove_unicode_chars(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', '', text)

import anthropic
anthropic_client = anthropic.Anthropic()


client = OpenAI()
# br_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# client for us-east-1
br_client = boto3.client("bedrock-runtime", region_name="us-west-2")

MessageRole = Literal["system", "user", "assistant"]


@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    response = client.completions.create(model=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=stop_strs,
    n=num_comps)
    
    # update token usage
    gpt_usage.completion_tokens += response.usage.completion_tokens
    gpt_usage.prompt_tokens += response.usage.prompt_tokens
    
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:

    if model == "o1-mini" or model == "o1-preview":
        m = [dataclasses.asdict(message) for message in messages]

        
        for message in m:
            if message["role"] == "system":
                message["role"] = "assistant"

            message['content'] = remove_unicode_chars(message['content'])
        
        response = client.chat.completions.create(model=model,
        messages=m,
        n=num_comps)

    else:
        response = client.chat.completions.create(model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.,
        frequency_penalty=0.,
        presence_penalty=0.,
        n=num_comps)

    print(response.choices[0].message.content)

    # update token usage
    gpt_usage.completion_tokens += response.usage.completion_tokens
    gpt_usage.prompt_tokens += response.usage.prompt_tokens
    
    
    
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def claude_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 1.0, #0.0,
    num_comps=1,
) -> Union[List[str], str]: 
    
    #pre-process messages
    system_prompts = None
    user_messages = []
    for message in messages:
        if message.role == 'system':
            if system_prompts is None:
                system_prompts = [{"text": message.content}]
            else:
                system_prompts.append([{"text": message.content}])
        
        else:
            msg = {"role": message.role,
                    "content": [{"text": message.content}]}
            user_messages.append(msg)
    
    response = br_client.converse(
        modelId=model,
        messages=user_messages,
        system=system_prompts,
        inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                        "topP": 0.9
                        },
    )

    # update token usage for Claude -- Pending
    gpt_usage.completion_tokens += response['usage']['outputTokens']
    gpt_usage.prompt_tokens += response['usage']['inputTokens']
    
    return response["output"]["message"]["content"][0]["text"] 


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def anthropic_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
    cache_hits: int = 0,
) -> Union[List[str], str]:
    
    #pre-process messages
    system_prompts = None
    user_messages = []
    for message in messages:
        if message.role == 'system':
            if system_prompts is None:
                # if cache_hits <= 4:
                #     system_prompts = [{"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}}]
                # else:
                system_prompts = [{"type": "text", "text": message.content}]
            else:
                # if cache_hits <= 4:
                #     system_prompts.append({"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}})
                # else:
                system_prompts.append({"type": "text", "text": message.content})
        
        else:
            # if cache_hits <= 4:
            #     msg = {"role": message.role,
            #             "content": [{"type": "text", "text": message.content, "cache_control": {"type": "ephemeral"}}]}
            # else:
            #     msg = {"role": message.role,
            #             "content": [{"type": "text", "text": message.content}]}
            
            msg = {"role": message.role,
                    "content": [{"type": "text", "text": message.content}]}
                
            user_messages.append(msg)
            

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        temperature=temperature,
        system=system_prompts,
        messages=user_messages,
    )

    # update token usage for Claude -- Pending
    gpt_usage.completion_tokens += response.usage.output_tokens
    gpt_usage.prompt_tokens += response.usage.input_tokens
    # gpt_usage.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
    # gpt_usage.cache_read_input_tokens += response.usage.cache_read_input_tokens
    
    print(f"cache hits: {cache_hits}")
    print(colored(f"API usage: {response.usage}", 'green'))
    
    return response.content[0].text

class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)


class ClaudeChat(ModelBase):
    """_summary_

    Args:
        ModelBase (_type_): _description_
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.is_chat = True
        self.cache_hit_ctr = 0 #log total hits for prompt caching

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        print("invoked")
        raise NotImplementedError

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return claude_chat(self.name, messages, max_tokens, temperature, num_comps)
        
        # # for anthropic client
        # self.cache_hit_ctr += 0
        # return anthropic_chat(self.name, messages, max_tokens, temperature, num_comps, cache_hits=self.cache_hit_ctr)

class Sonnet3(ClaudeChat):
    def __init__(self):
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        super().__init__(model_id)

class Sonnet35(ClaudeChat):
    def __init__(self):
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        super().__init__(model_id)


class Llama3_1_405B(ClaudeChat):
    def __init__(self):
        model_id = "meta.llama3-1-405b-instruct-v1:0"
        super().__init__(model_id)


class Llama3_1_70B(ClaudeChat):
    def __init__(self):
        model_id = "meta.llama3-1-70b-instruct-v1:0"
        super().__init__(model_id)
        
class Llama3_1_8B(ClaudeChat):
    def __init__(self):
        model_id = "meta.llama3-1-8b-instruct-v1:0"
        super().__init__(model_id)


class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT4o(GPTChat):
    """
    Added GPT4o
    """
    def __init__(self):
        super().__init__("gpt-4o")


class GPT4oMini(GPTChat):
    """
    Added GPT4o-mini
    """
    def __init__(self):
        super().__init__("gpt-4o-mini")


class o1(GPTChat):
    """
    GPT o1
    """
    def __init__(self):
        super().__init__("o1-preview")
        

class o1mini(GPTChat):
    """
    GPT o1
    """
    def __init__(self):
        super().__init__("o1-mini")


class GPT4turbo(GPTChat):
    """
    Added GPT4-turbo
    """
    def __init__(self):
        super().__init__("gpt-4-turbo")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")


class GPTDavinci(ModelBase):
    def __init__(self, model_name: str):
        self.name = model_name

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0, num_comps=1) -> Union[List[str], str]:
        return gpt_completion(self.name, prompt, max_tokens, stop_strs, temperature, num_comps)


class HFModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model_name: str, model, tokenizer, eos_token_id=None):
        self.name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        # NOTE: HF does not like temp of 0.0.
        if temperature < 0.0001:
            temperature = 0.0001

        prompt = self.prepare_prompt(messages)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=min(
                max_tokens, self.model.config.max_position_embeddings),
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self.eos_token_id,
            num_return_sequences=num_comps,
        )

        outs = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        assert isinstance(outs, list)
        for i, out in enumerate(outs):
            assert isinstance(out, str)
            outs[i] = self.extract_output(out)

        if len(outs) == 1:
            return outs[0]  # type: ignore
        else:
            return outs  # type: ignore

    def prepare_prompt(self, messages: List[Message]):
        raise NotImplementedError

    def extract_output(self, output: str) -> str:
        raise NotImplementedError


class StarChat(HFModelBase):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
        )
        super().__init__("starchat", model, tokenizer, eos_token_id=49155)

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += f"<|{message.role}|>\n{message.content}\n<|end|>\n"
            if i == len(messages) - 1:
                prompt += "<|assistant|>\n"

        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("<|assistant|>")[1]
        if out.endswith("<|end|>"):
            out = out[:-len("<|end|>")]

        return out


class CodeLlama(HFModelBase):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def __init__(self, version: Literal["34b", "13b", "7b"] = "34b"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            add_eos_token=True,
            add_bos_token=True,
            padding_side='left'
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"codellama/CodeLlama-{version}-Instruct-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        super().__init__("codellama", model, tokenizer)

    def prepare_prompt(self, messages: List[Message]):
        if messages[0].role != "system":
            messages = [
                Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            ] + messages
        messages = [
            Message(role=messages[1].role, content=self.B_SYS +
                    messages[0].content + self.E_SYS + messages[1].content)
        ] + messages[2:]
        assert all([msg.role == "user" for msg in messages[::2]]) and all(
            [msg.role == "assistant" for msg in messages[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        messages_tokens: List[int] = sum(
            [
                self.tokenizer.encode(
                    f"{self.B_INST} {(prompt.content).strip()} {self.E_INST} {(answer.content).strip()} ",
                )
                for prompt, answer in zip(
                    messages[::2],
                    messages[1::2],
                )
            ],
            [],
        )
        assert messages[-1].role == "user", f"Last message must be from user, got {messages[-1].role}"
        messages_tokens += self.tokenizer.encode(
            f"{self.B_INST} {(messages[-1].content).strip()} {self.E_INST}",
        )
        # remove eos token from last message
        messages_tokens = messages_tokens[:-1]
        import torch
        return torch.tensor([messages_tokens]).to(self.model.device)

    def extract_output(self, output: str) -> str:
        out = output.split("[/INST]")[-1].split("</s>")[0].strip()
        return out
