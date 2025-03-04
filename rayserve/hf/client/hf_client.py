# Usage: python hf_client.py <uuid> <text to be completed>
# Usage: python hf_client.py <profile> <name> <text to be completed>
# Ex: python hf_client.py default hf-biogpt "dog is "


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
    param4 = None
    if len(sys.argv)>4:
        param4 = sys.argv[4]
else:
    print("Please provide proper arguments\n, it should be : python <filename> <profile_name> <deployment_name> <text to be completed> \n IF it is published or under any other namespace, please give 4th argument as the namespace\n example : python hf_client.py dkubex hf-biogpt <text>\n OR \n example : python hf_client.py dkubex hf-biogpt <text> published \n OR \n example : python hf_client.py dkubex hf-biogpt <text> <namespace>")
    exit(1)

# get http url & token
ini_file = "/userdata/.d3x.ini"
config.read(ini_file)
url = config.get(param1,"url")
token = config.get(param1,"auth-token")


# get deployment details
headers = {'Authorization': token}
if len(sys.argv)>4:
    r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, params={"namespace": param4}, verify=False)
    deployment = r.json()['deployment']
else:
    r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, verify=False)
    deployment = r.json()['deployment']

# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"

# serving request
headers={'Authorization': SERVING_TOKEN}
resp = requests.post(SERVING_ENDPOINT, json={"prompt": param3}, headers=headers, verify=False)
print (resp.text)


