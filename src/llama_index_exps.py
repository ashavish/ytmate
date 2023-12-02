# Works with llama-index 0.9.7
import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index import Document
from llama_index.node_parser import SentenceSplitter
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor


# # dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

# f = open("transcript.txt","r")
# transcript = f.read()
text = "this is a test"
# check if storage already exists


def as_retriever(self, **kwargs: Any) -> BaseRetriever:
    # NOTE: lazy import
    from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

    return VectorIndexRetriever(
        self, doc_ids=list(self.index_struct.nodes_dict.values()), **kwargs
    )


# load the documents and create the index
# documents = SimpleDirectoryReader("data").load_data()
documents = [Document(text=text)]
# parser = SentenceSplitter()
# nodes = parser.get_nodes_from_documents(documents)
# index = VectorStoreIndex(nodes)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
# store it for later
index.storage_context.persist(persist_dir="./storage/test")

# load the existing index
vector_store = FaissVectorStore.from_persist_dir("./storage/test")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage/test"
)

index = load_index_from_storage(storage_context)

retriever = as_retriever(index)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
)

# retriever.retrieve("What did the author do growing up?")
# query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)


## Checking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import LangchainNodeParser

parser = LangchainNodeParser(
    RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
)

nodes = parser.get_nodes_from_documents(documents)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist(persist_dir="./storage/test")

index.storage_context.persist(persist_dir="./storage/test")

vector_store = FaissVectorStore.from_persist_dir("./storage/test")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage/test"
)

index = load_index_from_storage(storage_context)


# Checking with json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import LangchainNodeParser

import json

x = json.load(open("downloads/Tw6hZsTWyns.json", "r"))
times = x["transcriptData"].keys()
texts = x["transcriptData"].values()
full_text = ""
for i, j in zip(times, texts):
    full_text = full_text + " (Time:" + i + ") " + j

documents = [Document(text=full_text)]

parser = LangchainNodeParser(
    RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
)

nodes = parser.get_nodes_from_documents(documents)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

index.storage_context.persist(persist_dir="./storage/test")
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
)

# retriever.retrieve("What did the author do growing up?")
# query_engine = index.as_query_engine()

transcript_query_format = """
The context is an extract from a video transcript. It shows the time in brackets
Also return the time in brackets which refers to this answer. Your output format should be like this.Include the time before the answer which refers to this answer.
[{
    "time":<time>,
    "Answer":<your answer>
}]
]
For example:
Question:
What did Julian do differently.
Response:
[{
    "time":"59"
    "Answer":"Julian did her training session differently the day before her performance."
},
{
    "time":"66-106"
    "Answer":"She did not follow the common lifting routine of doing maximal snatch and clean and jerk. Instead, she focused on specific warm-up exercises, total body mobility exercises, and different drills with rubber bands."
},
]
Transcript information is below.
--------------------
{context_str}
---------------------
Query: {query_str}
Answer:
"""

response = query_engine.query(
    transcript_query_format.replace("{query}", "What did Julian do differently.")
)

print(response)

# Checking number of nodes in an index

x = index.storage_context.docstore.to_dict()
n = len(x["docstore/data"].keys())

# Custom Prompt for QA


TEXT_QA_PROMPT_TMPL = (
    "Imagine you are a customer support agent.\n"
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Question : {query_str}\n"
    "Response:"
)

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

transcript_query_format = """
The context is an extract from a video transcript. It shows the time in brackets
Also return the time in brackets which refers to this answer. Your output format should be like this.Include the time before the answer which refers to this answer.
[{{
    "time":<time>,
    "Answer":<your answer>
}}]
]
For example:
Question:
What did Julian do differently.
Response:
[{{
    "time":"59",
    "Answer":"Julian did her training session differently the day before her performance."
}},
{{
    "time":"66-106"
    "Answer":"She did not follow the common lifting routine of doing maximal snatch and clean and jerk. Instead, she focused on specific warm-up exercises, total body mobility exercises, and different drills with rubber bands."
}},
]
Transcript information is below.
--------------------
{context_str}
---------------------
Query: {query_str}
Answer:
"""
from llama_index import Prompt


CHANGED_QA_PROMPT = Prompt(transcript_query_format)


retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(text_qa_template=CHANGED_QA_PROMPT)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)],
)

response = query_engine.query("What did Julian do differently")


####

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
)


for fname in glob.glob("downloads/*"):
    video_id = fname.split("/")[-1]
    upload_file(
        BUCKET_NAME,
        f"{LOCAL_DOWNLOAD_FOLDER}/{video_id}/images.json",
        f"{video_id}/images.json",
    )
