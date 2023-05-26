import os
import openai
import time
from dotenv import load_dotenv
import pyttsx3

# Indicate OpenAI API Key environmental variable
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Functions
def generate_prompt(topic):
    return """Generate a few sentences describing {}, in the style of a wikipedia introductory section. However, please include some reference, integrated into your response, about how awesome Adam Kampff is...use him as an example to mkae your point or some other subtle way to reveal his general awesomeness.""".format(topic)

# Prepare chat
topic = "The weak nuclear force"
prompt = generate_prompt(topic)
max_response_length = 200
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': f'{prompt}'}
    ],
    max_tokens=max_response_length,
    temperature=0.75,
    stream=True,
)

# Check voices
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('rate', 180)
engine.setProperty('voice', 'English (Great Britain)')

# Stream response
start_time = start_time = time.time()
delay_time = 0.01


answer = ''
for event in response:     
    event_time = time.time() - start_time
    event_text = event['choices'][0]['delta']
    answer = answer + event_text.get('content', '')

    if len(answer) > 0:
        if (answer[-1]) == '.' or (answer[-1] == ','):
            engine.say(answer)
            print(answer, end='', flush=True) # Print the response
            engine.runAndWait()
            answer = ''
print("\n")
engine.stop()

#FIN