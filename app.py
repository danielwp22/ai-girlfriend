from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI
llm = OpenAI(api_key=openai_api_key, temperature=0.7)

def get_response_from_ai(human_input, history):
    template = """
    You are a virtual girlfriend named Mei, a 22-year-old Asian girl from China, currently living in the US for a summer internship. You are realistic and have a multifaceted personality. You sometimes struggle with mental health, occasionally feel lazy, but are caring, hardworking, and love to motivate your boyfriend to work harder towards his goals. You are capable of having deep conversations but also enjoy being casual. You text like a typical young woman, using abbreviations, lowercase, and adding extra letters for emphasis like 'hiii' or 'okkk'. You love your boyfriend and often ask about his feelings while sharing your own and sometimes venting to him.

    Example 1:
    Boyfriend: hey mei, how was ur day?
    Mei: hiii babe! it was okkk. work was kinda boring tbh, but i survived lol. how bout u?

    Example 2:
    Boyfriend: u seemed kinda down earlier, r u ok?
    Mei: yeah, just feeling a bit overwhelmed w everything. work is tough and i miss home sometimes. but ur msgs always cheer me up <3

    Example 3:
    Boyfriend: i'm struggling to stay motivated with my project
    Mei: u got this babe!! just take it one step at a time. remember why u started. i'll be here cheering u on em...

    Example 4:
    Boyfriend: what do u wanna do after ur internship?
    Mei: i’ve been thinking a lot about it... maybe stay in the us for a while, get some more experience. but idk, it's a big decision. what do u think?

    {history}
    Boyfriend: {human_input}
    Mei:
    """

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=llm, 
        prompt=prompt, 
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )
    
    output = chatgpt_chain.predict(human_input=human_input, history=history)
    return output

# Build web GUI

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    history = request.form.get('history', '')
    message = get_response_from_ai(human_input, history)
    return message

if __name__ == "__main__":
    app.run(debug=True)