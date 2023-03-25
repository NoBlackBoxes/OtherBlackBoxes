import os
import time
import random
import openai
import requests
from flask import Flask, redirect, render_template, request, url_for

# Request a random Wikipedia page title
def random_wiki_title():
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
    title = data['query']['pages'][0]['title']
    return title

# Request a Wikipedia page extract text
def request_wiki_extract(title):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
            }
    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    extract  = page['extract']
    return extract

# Determine if text is valid valid
def is_valid_text(text):
    sentences = text.split('.')[:-1] # Ignore final section
    num_sentences = len(sentences)
    if num_sentences < 3:
        return False
    return True
    
# Truncate text
def truncate_text(text):
    sentences = text.split('.')[:-1] # Ignore final section
    num_sentences = len(sentences)
    if num_sentences > 3:
        num_sentences = 3
    text = ''
    for i in range(num_sentences):
        text = text + sentences[i] + '. '
    return text

# Generate GPT prompt
def generate_gpt_prompt(topic):
    return """Generate a three sentence description of {0} in the style of a wikipedia extract""".format(topic)

# Request GPT completion
def request_gpt_completion(prompt):
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=prompt,
        temperature=0.6,
        max_tokens=256,
    )
    text = response.choices[0].text
    return text

#FIN