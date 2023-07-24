# Usage: python hf_client.py <uuid> <text to be completed>
# Ex: python hf_client.py 8165b659-e9ba-4ee7-b396-a10effe4818c "dog is "


import configparser
import requests
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = configparser.ConfigParser()


# Check if a parameter is provided
if len(sys.argv) > 2:
    param1 = sys.argv[1]
    param2 = sys.argv[2]
else:
    print("Please provide uuid.")
    exit(1)

# get http url & token
ini_file = "/tmp/.d3x.ini"
config.read(ini_file)
url = config.get("default","url")
token = config.get("default","auth-token")


# get deployment details
headers = {'Authorization': token}
r = requests.get(f"{url}/llm/api/deployments/{param1}", headers=headers, verify=False)
deployment = r.json()['deployment']

# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"

# serving request
headers={'Authorization': SERVING_TOKEN}
resp = requests.post(SERVING_ENDPOINT, json={"prompt": param2}, headers=headers, verify=False)
print (resp.text)

