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

# State
is_wiki_1 = True
num_correct = 0
num_wrong = 0

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():

    # Reset score counter
    num_correct = 0
    num_wrong = 0

    return redirect(url_for("select"))

# Specify select routes
@app.route("/select1", methods=("GET", "POST"))
def select1():
    global is_wiki_1, num_correct, num_wrong
    if is_wiki_1:
        num_correct += 1
        return redirect(url_for("correct"))
    else:
        num_wrong += 1
        return redirect(url_for("wrong"))

@app.route("/select2", methods=("GET", "POST"))
def select2():
    global is_wiki_1, num_correct, num_wrong
    if is_wiki_1:
        num_wrong += 1
        return redirect(url_for("wrong"))
    else:
        num_correct += 1
        return redirect(url_for("correct"))

@app.route("/wrong", methods=("GET", "POST"))
def wrong():
    return render_template("wrong.html"), {"Refresh": "0.25; url=select"}

@app.route("/correct", methods=("GET", "POST"))
def correct():
    return render_template("correct.html"), {"Refresh": "0.25; url=select"}

@app.route("/select", methods=("GET", "POST"))
def select():
    global is_wiki_1, num_correct, num_wrong

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
    print("\n\Title: {0}\n----\nGPT: {1}\n-----\nWiki: {2}\n\n{3} vs {4}".format(title, gpt_text, wiki_text, num_correct, num_wrong))

    # Randomize position
    if bool(random.getrandbits(1)):
        result1 = gpt_text
        result2 = wiki_text
        is_wiki_1 = False
    else:
        result1 = wiki_text
        result2 = gpt_text
        is_wiki_1 = True

    return render_template("index.html", title=title, result1=result1, result2=result2, num_correct=num_correct, num_trials=(num_correct+num_wrong))


#FIN