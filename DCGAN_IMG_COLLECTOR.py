#%% create scrape to get anime face raw image from web page konachan.net
import requests
from bs4 import BeautifulSoup
import os
import traceback
from tqdm import tqdm 

#%% create function to download image
def download(url, filename):
    if os.path.exists(filename):
        print('file exists!')
        return 
    try:
        r = requests.get(url, stream=True, timeout=60)# request url
        r.raise_for_status() # store http error 
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)
# %% create the imgs folder
if not os.path.exists('imgs'):
    os.makedirs('imgs')

# %%
start = 7992
end = 8000
for i in tqdm(range(start, end + 1), position=0):
    url = 'http://konachan.net/post?page=%d&tags=' % i
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup('img', class_="preview"):
        target_url =  img['src']
        filename = os.path.join('imgs', target_url.split('/')[-1])
        download(target_url, filename)
# %%

# %