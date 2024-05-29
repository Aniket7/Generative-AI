
# this python file is used to build the LLM application

# import the required packages
import json
from transformers import AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline


# load the configuration file
with open("config.json", 'r') as file:
	config_data = json.loads(file)

# get the required configuration parameters
model_name = config_data["model_name"]
context_length = config_data["context_length"]
max_new_tokens = config_data["max_new_tokens"]
seed = config_data["seed"]
temperature = config_data["temperature"]
repetition_penalty = config_data["repetition_penalty"]
stop = config_data["stop"]
top_k = config_data["top_k"]
top_p = config_data["top_p"]

# initialize the llm model object
llm_model = AutoModelForCausalLM.from_pretrained(model_name, model_type="mistral", context_length=context_length, max_new_tokens=max_new_tokens)

# configure prompts
template = config_data["template"]
message = ""
prompt = PromptTemplate(template=template, input_variable=["question"])
prompt_template = prompt.format(question=message)

# generate the text using llm
output = llm_model(prompt_template, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty, stop=stop, top_p=top_p, top_k=top_k)


print("This is the output -> ", output)