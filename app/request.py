import requests
import json


def send_test_event(query):
    url = "http://localhost:8000/query/"
    query_data = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(url=url, data=json.dumps(query_data), headers=headers)
    print(response.json()["answer"])
    return response

if __name__ == "__main__":
    query = "What are the bootstrapping and random forest algorithms?"
    send_test_event(query)