from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .generator_types import Generator
from .model import CodeLlama, \
                    ModelBase, \
                    GPT4, \
                    GPT4turbo, \
                    GPT4o, \
                    GPT4oMini, \
                    o1, o1mini, \
                    GPT35, \
                    StarChat, \
                    GPTDavinci, \
                    Sonnet3,\
                    Sonnet35, \
                    Llama3_1_405B, Llama3_1_70B, Llama3_1_8B


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        print("using GPT-4")
        return GPT4()
    elif model_name == "gpt-4o":
        print("using GPT-4o")
        return GPT4o()
    elif model_name == "gpt-4o-mini":
        print("using GPT-4o-mini")
        return GPT4oMini()
    elif model_name == "o1":
        print("using o1")
        return o1()
    elif model_name == "o1-mini":
        print("using o1-mini")
        return o1mini()
    elif model_name == "gpt-4-turbo":
        print("using GPT-4-Turbo")
        return GPT4turbo()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    
    # add support for Claude models
    elif model_name == "claude_3_sonnet":
        print("using Claude 3 Sonnet")
        return Sonnet3()
    
    elif model_name == "claude_35_sonnet":
        print("using Claude 3.5 Sonnet")
        return Sonnet35()

    elif model_name == "llama3_1_405b":
        print("using LLama 3.1 405B")
        return Llama3_1_405B()
    
    elif model_name == "llama3_1_70b":
        print("using LLama 3.1 70B")
        return Llama3_1_70B()
    
    elif model_name == "llama3_1_8b":
        print("using LLama 3.1 8B")
        return Llama3_1_8B()
    
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
