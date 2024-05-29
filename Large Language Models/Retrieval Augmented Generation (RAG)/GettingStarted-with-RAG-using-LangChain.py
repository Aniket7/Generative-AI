##### This code is used to build RAG application using LangChain framework 


# import the libraries
from tranformers import(
	AutoModelForCausalLM,
	pipeline,
	AutoTokenizer
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# input pdf
pdf_path = "sample.pdf"

# create an instance of the PyPDFLoader object where we pass the path to our file.
loader = PyPDFLoader(file_path=pdf_path)
# It returns an array consisting of Document objects, where each of these objects is a representation of one page of our file.
documents = loader.load()

# We don’t want to send a whole document as a context with our query to the LLM. initilize the split object
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# split the documents
docs = splitter.split_documents(documents)


# initilize the vector DB object and Load chunked documents into the FAISS index with embeddings algorithm
db = FAISS(docs, HuggingFaceEmbeddings=(model_name='sentence-transformers/all-mpnet-base-v2'))
# save the vector_db_model
db.save_local("vector_db_model")
# create the vector DB object using docs and embeddings algorithm
retriever = db.as_retriever()

# initilize model_name: download model of your choice from huggingface hub and save in your directory. For example
model_name='mistralai/Mistral-7B-Instruct-v0.1'

model_config = transformers.AutoConfig.from_pretrained(
    model_name,
)

# initilize tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# initilize the model object
model = AutoModelForCausalLM.from_pretrained(model_name)

# Build llm model text generation pipeline
text_gen_pipeline = pipeline(
		model = model,
		tokenizer = tokenizer,
		task = "text-generation",
		temperature = 0.2,
		repetition_penalty = 1.1,
		max_new_tokens = 100,
		return_full_text = True
		)
		

# initilize llm model
mistral_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

#Create PromptTemplate 
prompt_template = """
### [inst] INSTRUCTION: Answer the question 

{context}

### QUESTION
{question} [/inst]
"""

# initilize prompt
prompt = PromptTemplate(
	input_variable = ["context", "question"]
	template = prompt_template
	)

# initilize LLMChain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

Q = "QUESTION ?"

llm_chain.invoke(Q)

# initilize RAG_chain
RAG_chain = (
{"context":retriever, "question":RunnablePassThrough()}
	| llm_chain
	)

Q = "QUESTION ?"

# get the result from RAG_chain
result = RAG_chain.invoke(Q)

print(result["context"])

print(result["text"])