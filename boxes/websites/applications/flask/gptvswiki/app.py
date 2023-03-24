import os
import openai
import random
import time
import requests
from flask import Flask, redirect, render_template, request, url_for

# Create Flask app
random.seed(time.time())
app = Flask(__name__)

# Indicate OpenAI API Key environmental variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":

        # Request a random Wikipedia page
        url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'info',
            'generator': 'random',
            "formatversion": "2",
	        "grnnamespace": "0",
	        "grnlimit": "1"
            }
        response = requests.get(url, params=params)
        data = response.json()
        topic = data['query']['pages'][0]['title']

        # Wiki
        url = 'https://en.wikipedia.org/w/api.php'
        params = {
                'action': 'query',
                'format': 'json',
                'titles': topic,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                }
        wiki_response = requests.get(url, params=params)
        wiki_data = wiki_response.json()
        wiki_page = next(iter(wiki_data['query']['pages'].values()))
        wiki_text  = wiki_page['extract']
        print(wiki_text)

        # Truncate Wiki response
        sentences = wiki_text.split('.')[:-1]
        print(sentences)
        num_sentences = len(sentences)
        if(num_sentences > 3):
            sentences = sentences[:3]
            num_sentences = 3
        print(num_sentences)

        # Re-assemble Wiki response
        wiki_text = ''
        for s in sentences:
            wiki_text = wiki_text + s + '. '

        # GPT
        prompt = generate_prompt(topic, num_sentences)
        print(prompt)
        gpt_response = openai.Completion.create(
            model="text-curie-001",
            prompt=prompt,
            temperature=0.6,
            max_tokens=256,
        )
        gpt_text = gpt_response.choices[0].text

        # Report
        print("Topic: {0}\n----\nGPT: {1}\n-----\nWiki: {2}".format(topic, gpt_text, wiki_text))

        # Randomize position
        if bool(random.getrandbits(1)):
            result1 = gpt_text
            result2 = wiki_text
        else:
            result1 = wiki_text
            result2 = gpt_text

        return redirect(url_for("index", topic=topic, result1=result1, result2=result2))

    topic = request.args.get("topic")
    result1 = request.args.get("result1")
    result2 = request.args.get("result2")

    return render_template("index.html", topic=topic, result1=result1, result2=result2)

def generate_prompt(topic, num_sentences):
    return """Generate {0} sentences, not in a list, describing {1} in the style of a wikipedia extract""".format(num_sentences, topic)

#FIN