import requests

#subject = 'Python (programming language)'
subject = 'Silicon'
url = 'https://en.wikipedia.org/w/api.php'
params = {
        'action': 'query',
        'format': 'json',
        'titles': subject,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
        }
 
response = requests.get(url, params=params)
data = response.json()
 
page = next(iter(data['query']['pages'].values()))
text = page['extract'] 
sentences = text.split('.')
num_sentences = len(sentences)

print(text)

# Get images
subject = 'Silicon'
url = 'https://en.wikipedia.org/w/api.php'
params = {
        'action': 'query',
        'format': 'json',
        'titles': subject,
        'prop': 'images'
        }
 
response = requests.get(url, params=params)
data = response.json()
 
page = next(iter(data['query']['pages'].values()))
print(page['extract'][:])

# Get random
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
print(topic)



#FIN