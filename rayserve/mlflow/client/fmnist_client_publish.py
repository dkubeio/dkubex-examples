# Usage: python fminst_client.py <profile> <name> <image path>
# Ex: python fmnist_client.py default fashion-mnist-deploy ./images/pull-over.pn
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
    print("Please provide name.")
    exit(1)

# get http url & token
ini_file = "/userdata/.d3x.ini"
config.read(ini_file)
url = config.get(param1,"url")
token = config.get(param1,"auth-token")


# get deployment details
headers = {'Authorization': token}
if len(sys.argv)>4:
    if param4 == "published":
        r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, params={"namespace": param4}, verify=False)
        deployment = r.json()['deployment']
    else:
        r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, params={"namespace": param4} ,verify=False)
        deployment = r.json()['deployment']
else:
    r = requests.get(f"{url}/llm/api/deployments/{param2}", headers=headers, verify=False)
    deployment = r.json()['deployment']
    
# get serving details
SERVING_TOKEN = deployment['serving_token']
SERVING_ENDPOINT = f"{url}{deployment['endpoint']}"
IMAGE_PATH =  param3 #"/home/<user>/workspaces/default-workspace/ray/images/pull-over.png"

# convert image to bytes
with open(IMAGE_PATH, "rb") as image:
   f = image.read()
   image_bytes = bytearray(f)

# serving request
headers={'Authorization': SERVING_TOKEN}
resp = requests.post(SERVING_ENDPOINT, data=image_bytes, headers=headers, verify=False)
print (resp.json())


