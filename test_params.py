import requests
url = "https://hf-mirror.com/api/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
r = requests.get(url, timeout=10)
print(r.status_code)