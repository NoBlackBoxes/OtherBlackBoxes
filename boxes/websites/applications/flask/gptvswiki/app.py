import os
import time
import random
import openai
import requests
from flask import Flask, redirect, render_template, request, url_for
import utilities as utl

# Seed random
random.seed(time.time())

# Create Flask app
app = Flask(__name__)

# Indicate OpenAI API Key environmental variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():

    # Retrieve a valid Wikipedia extract
    while True:
        title = utl.random_wiki_title()
        extract = utl.request_wiki_extract(title)
        if utl.is_valid_text(extract):
            break

    # Truncate Wikipedia extract
    wiki_text = utl.truncate_text(extract)

    # Generate GPT prompt
    prompt = utl.generate_gpt_prompt(title)

    # Request GPT completetion
    gpt_text = utl.request_gpt_completion(prompt)

    # Truncate GPT extract
    gpt_text = utl.truncate_text(gpt_text)

    # Report
    print("\n\Title: {0}\n----\nGPT: {1}\n-----\nWiki: {2}\n\n".format(title, gpt_text, wiki_text))

    # Randomize position
    if bool(random.getrandbits(1)):
        result1 = gpt_text
        result2 = wiki_text
    else:
        result1 = wiki_text
        result2 = gpt_text

    return render_template("index.html", title=title, result1=result1, result2=result2)

#FIN