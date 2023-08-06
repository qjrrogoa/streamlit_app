#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch import cuda, bfloat16

from transformers import BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

import streamlit as st
import requests
from streamlit_chat import message

from huggingface_hub import login

login(token='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')


# In[2]:


'''
LoRA parameter를 가져오고 llama2에 연결합니다.
'''
peft_model_id = "RAIJAY/albatross_classifier"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

def evaluate_model(model, tokenizer, device, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=128):
    prompt_instruction = "### Instruction:\n{}\n" + (f"### Input:\n{input}\n" if input else "") + "### Response:\n"
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n" + prompt_instruction.format(instruction)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_length=max_new_tokens + inputs["input_ids"].shape[1]
        )
    
    output = tokenizer.decode(generation_output[0])
    return output.split("### Response:")[1].strip()

 
st.header("🦜Albatross")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

    
with st.form('form', clear_on_submit=True):
    user_input = st.text_area('You: ', '')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    output = evaluate_model(model, tokenizer, 'cuda', instruction="Predict FOMC' stance toward Interest Rate Dicision (Hawkish/Neutral/Dovish)", input=fomc)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if 'generated' in st.session_state:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))

