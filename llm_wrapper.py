from config import Config
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class Llm_wrapper(object) :
    def __init__(self, conf : Config):
        super().__init__()
        self.conf = conf
        self.llm = None
        self.embd_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=conf.embd_device)

    def embed(self, inputs):
        output = self.embd_model.encode([inputs])[0]
        return output

    def query(self, inputs, max_tokens, temperature, seed):
        # User external llama cpp server
        api_url = f"{self.conf.external_llama_cpp_url}/completion"
        in_data = {"prompt": inputs, "n_predict": max_tokens, "seed": seed,
                   "temperature": temperature, "stream" : True, "stop" : ["[/INST]"]}

        headers = {"Content-Type": "application/json"}
        if self.conf.external_llama_cpp_api_key is not None:
            headers["Authorization"] = f"Bearer {self.conf.external_llama_cpp_api_key}"
        response = requests.post(api_url, data=json.dumps(in_data), headers=headers, stream=True)

        for line in response.iter_lines():

            # filter out keep-alive new lines
            if line:
                decoded_line = line.decode('utf-8').replace("data: ", "")
                j_str = json.loads(decoded_line)
                if j_str['stop'] == "true" :
                    print("-- STOP --")
                    return
                # print(json.loads(decoded_line))
                yield j_str['content']



