#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import pinecone
import os
from transformers import BartTokenizer, BartForConditionalGeneration


# In[4]:


#read in data
parquet_file = "/Users/felix/Downloads/archive 2/"
df = pd.read_parquet(parquet_file, engine='pyarrow')
#df.head()


# In[5]:


#drop unnecessary columns
df = df.drop(columns=["url", "title"])
df.head()


# In[6]:


#check shape
df.shape


# In[7]:


#use gpu if can
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#load retriever model from huggingface models
retriever = SentenceTransformer("all-mpnet-base-v2")


# In[8]:


#check retriever dimension
retriever.get_sentence_embedding_dimension()


# In[9]:


#initialize pinecone
api_key = os.getenv("INSERT_YOUR_API_KEY") or "INSERT_YOUR_API_KEY"
env = os.getenv("gcp-starter") or "gcp-starter"

pinecone.init(api_key = api_key, environment = env)
#confirm connection
pinecone.whoami()


# In[10]:


#initialize index
index_name = 'sci-qa'

#check if index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension = retriever.get_sentence_embedding_dimension(),  #dimensionality of embeddings, 768
        metric = 'cosine',
    )
#connect to index
index = pinecone.Index(index_name)
#check index stats
index.describe_index_stats()


# In[128]:


#start batches to pinecone
batch_size = 100

#loop to send in batches of our data to pinecone
for batch_start in range(0, len(df), batch_size):
    #end of batch
    batch_end = min(batch_start+batch_size, len(df))
    #extract batches of data that need embeddings
    data_for_emb = df.iloc[batch_start:batch_end]["text"].tolist()
    #extract batches of data that will be our metadata
    metadata = df.iloc[batch_start:batch_end].to_dict(orient="records")
    #embeddings for batch
    embeddings = retriever.encode(data_for_emb).tolist()
    #create unique ids
    ids = [f"{idx}" for idx in range(batch_start, batch_end)]
    #add all to upsert list
    to_upsert = list(zip(ids, embeddings, metadata))
    #upsert into pinecone
    try:
    #try/except in case metadata exceeds limit of pinecone
        upserting = index.upsert(vectors= to_upsert)
    except:
    #see what batch # there was a metadata exceed limit error
        print("Metadata exceed limit " + str(batch_start))

#check we have all our vectors
index.describe_index_stats()


# In[11]:


#load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)


# In[12]:


#helper function to query pinecone
def query_pinecone(query, top_k):
    #generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    #search pinecone index for context text with the answer
    xc = index.query(xq, top_k= top_k, include_metadata=True)
    return xc


# In[13]:


#helper function to format question/context so BART can read
def format_query(query, context):
    #extract text from pinecone search result, add <P> tag
    context = [f"<P> {m['metadata']['text']}" for m in context]
    #concatinate all context passages
    context = " ".join(context)
    #contcatinate query and context text
    query = f"question: {query} context: {context}"
    return query


# In[18]:


#helper function to generate answser using BART ELI5
def generate_answer(query):
    #tokenize query to get input_ids
    inputs = tokenizer([query], truncation = True, max_length= 1024, return_tensors= "pt").to(device)
    #use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams= 2, min_length= 20, max_length = 40)
    #use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens = True, clean_up_tokenization_spaces = False)[0]
    return answer


# In[19]:


query = input("What's your science question?\n")
context = query_pinecone(query, top_k= 3)
query = format_query(query, context["matches"])
generate_answer(query)


# In[ ]:




