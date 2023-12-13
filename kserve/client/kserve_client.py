# Usage: python kserve_client.py <profile> <name> <input json file name>
# Ex: python kserve_client.py default sklearnv1 input.json


import configparser
import requests
import sys
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

config = configparser.ConfigParser()


# Check if a parameter is provided
if len(sys.argv) > 3:
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    param3 = sys.argv[3]
else:
    print("Please provide name.")
    exit(1)

# get http url & token
ini_file = "/userdata/.d3x.ini"
config.read(ini_file)
url = config.get(param1,"url")
token = config.get(param1,"auth-token")


# get deployment details
headers = {'Authorization': token}
r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, verify=False)
deployment = r.json()['deployment']

# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"

# serving request
headers={'Authorization': SERVING_TOKEN, 'Content-Type': 'application/json'}
resp = requests.post(SERVING_ENDPOINT, data=open(param3, 'rb'), headers=headers, verify=False)
print (resp.text)

