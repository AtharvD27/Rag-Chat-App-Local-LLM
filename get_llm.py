from langchain_community.llms import LlamaCpp
import yaml


def get_local_llm(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm_config = config.get("llm", {})

    return LlamaCpp(
        model_path=llm_config.get("local_model_path", "./models/mistral.gguf"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 512),
        top_p=llm_config.get("top_p", 0.95),
        n_ctx=llm_config.get("n_ctx", 2048),
        n_threads=llm_config.get("n_threads", 4),
        verbose=False,
    )
