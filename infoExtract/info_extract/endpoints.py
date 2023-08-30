from predibase import PredibaseClient


class LLMEndpoint:
    def __init__(self, **kwargs):
        pass

    def hit(self, input_text):
        pass


class PredibaseLLMEndpoint(LLMEndpoint):
    def __init__(self, predibase_client: PredibaseClient, model_name: str = "llama-2-13b"):
        super().__init__(predibase_client=predibase_client, model_name=model_name)
        self.predibase_client = predibase_client
        self.model_name = model_name

    def hit(self, input_text):
        input_text = input_text.replace("'", "")
        options = {"max_new_tokens": 512, "temperature": 0.0}
        result = self.predibase_client.prompt(input_text, self.model_name, options=options)
        resp = result.response[0].strip()
        return resp


def get_llm_endpoint(model_provider, **kwargs):
    if model_provider == "predibase":
        return PredibaseLLMEndpoint(**kwargs)
    else:
        raise ValueError("Invalid LLM provider")
