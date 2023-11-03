from predibase import PredibaseClient
import re

pc = PredibaseClient()


class ChatAgent:
    """
    ChatAgent is a wrapper around the LLM model that keeps track of the chat history.

    Example:
        >>> llm = pc.LLM("pb://deployments/llama-2-13b-chat")
        >>> agent = ChatAgent(llm)
        >>> agent.chat("Hello, how are you?")
        'I am fine, how are you?'
    """

    def __init__(self, target_llm, adapter=True):
        base_model = pc.LLM("pb://deployments/llama-2-7b")
        default_model = pc.LLM("pb://deployments/llama-2-13b-chat")

        if adapter:
            self._llm = base_model.with_adapter(target_llm)
        else:
            self._llm = default_model

        self.internal_chat_history = ["### Context: You are a helpful, detailed, and polite artificial intelligence "
                                      "assistant. Your name is Lisa. Your answers are clear and suitable for a "
                                      "professional environment. Only provide one response when prompted with '### "
                                      "Assistant'"]
        self.external_chat_history = []

    @staticmethod
    def cleanse_response(llm_response):
        """
        Function to post-process the response from the LLM.
        :param llm_response: response from the LLM
        :return: cleansed response
        """
        # Remove User prompt
        cleaned_response = llm_response.replace("### User: ", "")

        # Remove Assistant prompt
        cleaned_response = cleaned_response.replace("### Assistant: ", "")

        # Remove leading newlines
        cleaned_response = cleaned_response.lstrip("\n")

        return cleaned_response

    def chat_completion(self, prompt, max_tokens=512, temperature=0.1):
        """
        This function calls a Predibase hosted LLM and responds to the prompt.
        :param prompt: Input from the user
        :param max_tokens: Max tokens to generate
        :param temperature: Randomness of the generated text
        :return: The generated response from the LLM.
        """
        # Add user input to external chat history for rendering
        self.external_chat_history.append({"role": "user", "content": prompt})

        # Add user input to internal chat history for generation
        self.internal_chat_history.append("### User: " + prompt)
        self.internal_chat_history.append("### Assistant: ")

        # Combine chat history into a single string to send to the LLM for generation
        chat_input = "\n".join(self.internal_chat_history)

        # Generate response from LLM
        response = self._llm.prompt(
            data=chat_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        print("BEFORE CLEANSE: ", response.response)
        cleaned_response = self.cleanse_response(response.response)
        print("AFTER CLEANSE: ", cleaned_response)

        # Add response to external chat history for rendering
        self.external_chat_history.append({"role": "assistant", "content": cleaned_response})

        # Add response to internal chat history for generation
        self.internal_chat_history[-1] = self.internal_chat_history[-1] + cleaned_response
        return cleaned_response

    def intent_classification(self, prompt):
        """
        This function calls a Predibase hosted LLM and classifies the intent of the prompt.
        :param prompt: Input from user to classify intent with
        :return: The classified intent from the LLM.
        """
        prompt_template = f"""You are an AI assistant and your task is to classify the intent of the following message. 
        You can choose between the following options: ["Cancel Shift", "Reschedule Shift", "Request Info", "Unknown"]. 
        If you are unsure, please respond with "Unknown". 
        Respond only with the intent. 
        Here are some examples:
        
        Example 1:
        ### Message: Hello, I would like to cancel my shift!
        Cancel Shift
                    
        Example 2:
        ### Message: Hello, I would like to reschedule my shift!
        Reschedule Shift
                    
        Example 3:
        ### Message: Gabagooooool!
        Unknown

        What is the intent of the following message: "{prompt}"
        """

        classified_intent = self._llm.prompt(
            data=prompt_template,
            max_new_tokens=128,
            temperature=0.1,
        )

        extracted_intent = re.findall(
            r'(Reschedule Shift|Cancel Shift|Request Info|Unknown)',
            classified_intent.response
        )

        if extracted_intent:
            return extracted_intent[0]
        else:
            print("ERROR: Could not extract intent from response: ", classified_intent.response)
            return "Unknown"
