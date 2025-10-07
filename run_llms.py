import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from surrealdb import RecordID,Surreal
from tqdm import tqdm


# from surreal import create_conn,add_or_update

load_dotenv()
#. Please generate the new LLM responses using GPT 4.1, o3, the latest Gemini model, the latest DeepSeek and the latest Llama model ideally
final_sample=pd.read_csv('/home/pmwaniki/Dropbox/others/Ambrose/Sync study/Benchmark/data/final_sample.csv')

db = Surreal('https://aws.datahubweb.com:6000')
db.signin({'username': "root", 'password': os.getenv("SURREALDB_PASSWORD")})
db.use("llm_study", "benchmark")


def hf_endpoint(repo_id, task="text-generation", max_new_tokens=512, do_sample=False, repetition_penalty=1.03):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task=task,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        huggingfacehub_api_token=os.getenv("HG_TOKEN"),
    )
gemma_endpoint=HuggingFaceEndpoint(
        endpoint_url="https://fcyfavjiuc9rp353.us-east-1.aws.endpoints.huggingface.cloud",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=os.getenv("HG_TOKEN"),
    )

model_dict={
    'llama4-maverick':{
        'function':ChatHuggingFace,
        'args':dict(llm=hf_endpoint(repo_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct"))
    },
    # 'gemini-2.0-flash':{
    #     'function':ChatGoogleGenerativeAI,
    #     'args':dict(model="gemini-2.0-flash")
    # },
    'gemini-2.5-flash':{
        'function':ChatGoogleGenerativeAI,
        'args':dict(model="gemini-2.5-flash")
    },
    'deepSeek-r1':{
        'function':ChatHuggingFace,
        'args':dict(llm=hf_endpoint(repo_id="deepseek-ai/DeepSeek-R1"))
    },
    'medgemma':{
        'function':ChatHuggingFace,
        'args':dict(llm=gemma_endpoint)
    },
    'gpt-4.1':{
        'function':ChatOpenAI,
        'args':dict(model="gpt-4.1")
    },
    'o3':{
        'function':ChatOpenAI,
        'args':dict(model="o3")
    },
    'llama3.2:3b':{
        'function':ChatOllama,
        'args':dict(model="llama3.2:3b")
    },
    'gemma3:4b':{
        'function':ChatOllama,
        'args':dict(model="gemma3:4b")
    },
    'deepseek-r1:8b':{
        'function':ChatOllama,
        'args':dict(model="deepseek-r1:8b")
    },
    'gpt-4o':{
        'function':ChatOpenAI,
        'args':dict(model="gpt-4o", model_provider="openai")
    },
    'o4-mini':{
        'function':ChatOpenAI,
        'args':dict(model="o4-mini")
    },


}



system_prompts={
    'prompt1':"""
    you (the LLM) are a professor of primary healthcare in Kenya. 
    You will be receiving queries from nurses as part of an experiment, and you should answer the questions to the best of your abilities, and use language and guidance that is contextually appropriate and relevant to the practice of medicine by a community nurse in Kenya.
     Leverage local guidelines where appropriate to inform your responses. The next thing you receive will be the nurse’s communication.
    """
}

example_questions=final_sample


model_def=[
    # 'gemma3:4b',
    # 'llama3.2:3b',
    # 'deepseek-r1:8b',
    # 'llama4-maverick',
    'gpt-4.1',
    # 'gemini-2.0-flash',
    'gemini-2.5-flash',
    'deepSeek-r1',
    'o3',
    # 'gpt-4o',
    'medgemma'
]


output_p=[]
rerun=False
for model_str in model_def:
    model_ = model_dict[model_str]['function'](**model_dict[model_str]['args'])
    print(f"Running inference for model {model_str}")

    for sys_p_key,sys_p in system_prompts.items():
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", sys_p), ("user", "{text}")]
        )
        for _,row in tqdm(example_questions.iterrows(),total=example_questions.shape[0]):
            result_id=f"{model_str}__{row['Master_Index']}__{sys_p_key}"
            if not rerun:
                r_=db.select(RecordID('prompt',result_id))
                if r_ is not None:
                    continue
            prompt_ = prompt_template.invoke({"text": row['Scenario']})
            result_ = model_.invoke(prompt_)
            result_dict={'id':result_id,'index':row['Master_Index'],'model':model_str,'system_prompt': sys_p_key,'user_prompt': row['Scenario'],
                             'response': result_.content,'now':datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            output_p.append(result_dict)
            try:
                db.upsert(RecordID('prompt',result_id),result_dict)
            except:
                db = Surreal('https://aws.datahubweb.com:6000')
                db.signin({'username': "root", 'password': os.getenv("SURREALDB_PASSWORD")})
                db.use("llm_study", "benchmark")
                db.upsert(RecordID('prompt', result_id), result_dict)



result_data=pd.DataFrame(output_p)

result_data.to_csv('/home/pmwaniki/Dropbox/others/Ambrose/Sync study/Benchmark/results/prompt_output.csv',index=False)

# db.query("select * from prompt")
model_dict.keys()
