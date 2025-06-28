
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from prompts import prompt_dict


#. Please generate the new LLM responses using GPT 4.1, o3, the latest Gemini model, the latest DeepSeek and the latest Llama model ideally

model_dict={
    'llama3.1':{
        'function':OllamaLLM,
        'args':dict(model="llama3.1:latest")
    },
    'deepseek-r1':{
        'function':OllamaLLM,
        'args':dict(model="deepseek-r1:7b")
    }
}



m='deepseek-r1'
t='prompt1'

test_question="""I am a nurse with 14 years of experience in General nursing working in a Dispensaries and Private Clinics in Kiambu county in Kenya. 
I have a mother at first visit. She's 30 years old.
She's part 4+0, gravida 5. After ANC Profile, the HB was 5.5. But everything else was okay. Weight also okay. Now, she has 34 weeks for gestation.
Now, my question is, what should I give? So, yeah, the question is? What should I give? IFAS, Aferon or iron sucrose. And also, I wanted to know why the patient didn't come to ANC. What could be the reason?
"""

model=model_dict[m]['function'](**model_dict[m]['args'])
prompt_template = ChatPromptTemplate.from_messages(
    [("system", prompt_dict[t]), ("user", "{text}")]
)
prompt = prompt_template.invoke({"text": test_question})

result=model.invoke(prompt)
print(result)