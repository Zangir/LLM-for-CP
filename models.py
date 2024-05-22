import re
import os
import google.generativeai as genai
from openai import OpenAI
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import time

###### MODELS ###### 

### GPT-N ###
def create_openai_model():
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    model = OpenAI(api_key=OPENAI_API_KEY)
    return model

def predict_openai_model(model, model_name, inp): # model_name: gpt-3.5-turbo-0613, gpt-4-0613
    response = model.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": "You are a problem solver."},
        {"role": "user", "content": inp}
      ],
      temperature=0
    )
    out_pred = response.choices[0].message.content
    
    return out_pred

def predict_openai_ci_model(model_name, inp): # model_name: gpt-4-turbo-preview
    ### Code Interpreter
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a problem solver. Write and run code to answer questions if needed.",
        tools=[{"type": "code_interpreter"}],
        model=model_name,
    )
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=inp
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1) # Wait for 1 second
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
    out_pred = ""
    if run.status == 'completed': 
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        try:
            out_pred = messages.data[0].content[0].text.value
        except:
            pass
    
    return out_pred

### PALM/GEMINI-N ###
def create_google_model(model_name): # model_name: gemini-pro
    API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name) 
    return model

def predict_google_model(model, inp):
    response = model.generate_content(inp)
    out_pred = response.text
    return out_pred

### HUGGINGFACE ###
def create_huggingface_model(model_name): # model_name: meta-llama/Llama-2-70b-chat-hf, meta-llama/Llama-2-7b-chat-hf
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    return model, tokenizer

def predict_huggingface_model(model, tokenizer, inp):
    device = "cuda"
    messages = [
        {"role": "user", "content": inp},
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    out_pred = decoded[0]
    
    return out_pred

class llm_function():
    
    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            self.model = create_openai_model()
        elif model_type == "google":
            self.model = create_google_model(model_name)
        elif model_type == "huggingface":
            self.model, self.tokenizer = create_huggingface_model(model_name)

    def predict(self, inp):
            
        if self.model_type == "openai":
            out = predict_openai_ci_model(self.model_name, inp)
        elif self.model_type == "google":
            out = predict_google_model(self.model, inp)
        elif self.model_type == "huggingface":
            out = predict_huggingface_model(self.model, self.tokenizer, inp)

        return out

def get_cp_solution(out_pred):
    solution = []
    ints = re.findall(r'\d+', out_pred)
    for elem in ints:
        solution.append(int(elem)-1)
    return solution