import os
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

repo_path = "storage/"

endPoint = os.getenv('endPoint')


# Point to the local server
client = OpenAI(base_url=endPoint, api_key="not-needed")

# Load documents
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".py"],
    exclude=["**/non-utf8-encoding.py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
)

documents = loader.load()

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)


# Prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

question = "what is the 41 presidents name?"

context = texts


# Render the QA prompt with context and question
rendered_prompt = QA_CHAIN_PROMPT.format(
    context = context,  # Use the context variable here
    question = question
)

# Create a message for the OpenAI client to process
messages = [
    {"role": "system", "content": rendered_prompt}
]

# Use the OpenAI client to generate the answer
completion = client.chat.completions.create(
  model="local-model",  # Adjust the model as needed
  messages=messages,
  temperature=0.7,
)

# Print the response
print(completion.choices[0].message)
