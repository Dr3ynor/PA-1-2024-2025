import requests

import multiprocessing as mp

# používat multiprocesing a ne multithreading

def say_hello(_):
    pid = mp.current_process().pid
    print(f"Hello from {pid}")
    return pid

def download(id):
    url = f"https://name-service-phi.vercel.app/api/v1/names/{id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"file{id}.txt", "wb") as f:
            f.write(response.content)
        print(f"Downloaded file{id}.txt")
    else:
        print(f"Failed to download file{id}.txt")
        
with mp.Pool(5) as p:
    result = p.map(say_hello, range(25))
    result = p.map(download, range(25))
    