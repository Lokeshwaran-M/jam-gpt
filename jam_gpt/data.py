import json
import torch
from . import config


class Data:
    """ just a data preprocessor """
    def __init__(self) -> None:
        pass

    @classmethod
    def get(cls, path: str) -> str:
        """
        text data file -> string data

        """
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()
        return text_data

    @classmethod
    def set(cls, path: str, data: str) -> None:
        """
        string data -> text data file
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
            print("writen data : ", len(data))

    @classmethod
    def train_test_split(cls, data, split_percent: int = 90):
        """
        split the data into train ans test based on split percentage
        """
         
        tensor_data = torch.tensor(data, dtype=torch.long, device=config.device)
        n = int((split_percent/100)*len(data))
        train_data = tensor_data[:n]
        test_data = tensor_data[n:]

        return [train_data, test_data]
    
    @classmethod
    def chat_formater(cls,context=None, prompt=None, response=None):
        """Creates a JSON object from the given context, prompt, and output.

        Args:
            context: A string containing the context
            prompt: A string containing the prompt
            Response: A string containing the Response

        Returns:
            A JSON object containing the context, prompt, and Response
        """
        
        if response :
            data = f"### context:\n{context}\n\n### prompt:\n{prompt}\n\n### response:\n{response}\n [eos] \n"
        elif context and prompt :
            data = f"### context:\n{context}\n\n### prompt:\n{prompt}\n\n### response:\n"
        elif not context :
            data = f"### prompt:\n{prompt}\n\n### response:\n"

        return data
    
    @classmethod
    def chat_JsonToTxt(cls,path_json,path_txt):

        # Read data from the JSON file
        with open(path_json, 'r') as json_file:
            data = json.load(json_file)

        # Create a text file to save the data
        with open(path_txt, 'w') as txt_file:
            # Iterate through the data and write context, prompt, and response to the text file
            for chat in data:
                context = chat['context']
                prompt = chat['prompt']
                response = chat['response']

                # Write to the text file
                formated_chat = cls.chat_formater(context, prompt, response)
                txt_file.write(formated_chat)
                txt_file.write('\n')  # Add an empty line to separate entries




