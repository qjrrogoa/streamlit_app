# %%
import streamlit as st
import requests
from streamlit_chat import message
 
def generate_response(prompt):
    completions = openai.Completion.create (
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        stop=None,
        temperature=0,
        top_p=1,
    )
 
    message = completions["choices"][0]["text"].replace("\n", "")
    return message
 
st.header("🤖DM's ChatGPT 3.5")

 
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

    
with st.form('form', clear_on_submit=True):
    user_input = st.text_area('You: ', '')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if 'generated' in st.session_state:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))




