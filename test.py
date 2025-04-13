import requests

url = 'http://127.0.0.1:5000/upload'
file_path = r'C:\Users\God-user\Desktop\Notebook Testing\uploads\titanic.ipynb'

with open(file_path, 'rb') as f:
    response = requests.post(url, files={'file': f})

print(response.json())
