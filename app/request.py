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
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print(f"Full Response Headers: {response.headers}")


if __name__ == "__main__":
    query = "Can you explain how to use FastAPI?"
    send_test_event(query)