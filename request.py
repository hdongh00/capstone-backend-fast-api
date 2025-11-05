import requests

class Axios:
    def __init__(self, token, content):
        self.url = "http://localhost:8080"
        self.headers = {"Authorization": "Bearer "+token,"Content-Type": content}

    def post(self, url, data):
        response = requests.post(url=self.url+url, data=data, headers=self.headers)
        print(response)