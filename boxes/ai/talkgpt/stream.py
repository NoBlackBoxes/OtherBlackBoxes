import os
import openai
import time
from dotenv import load_dotenv

# Indicate OpenAI API Key environmental variable
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Functions
def generate_prompt(topic):
    return """Generate a few sentences describing {}, in the style of a wikipedia introductory section""".format(topic)

# Prepare chat
topic = "proton"
prompt = generate_prompt(topic)
max_response_length = 200
response = openai.ChatCompletion.create(
    # CHATPG GPT API REQQUEST
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': f'{prompt}'}
    ],
    max_tokens=max_response_length,
    temperature=0.75,
    stream=True,
)

# Stream response
start_time = start_time = time.time()
delay_time = 0.01
for event in response:     
    event_time = time.time() - start_time
    event_text = event['choices'][0]['delta']
    answer = event_text.get('content', '')
    time.sleep(delay_time)

    print(answer, end='', flush=True) # Print the response

#FIN