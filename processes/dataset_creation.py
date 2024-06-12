import os
import urllib.request
from bs4 import BeautifulSoup
import re


def data_fetch_from_url():
    # URL of the webpage to extract data from
    url = "https://en.wikipedia.org/wiki/Mercedes-Benz"

    # Fetch the webpage
    response = urllib.request.urlopen(url)
    webContent = response.read()
    print("read complete from url")

    # Parse the webpage content
    soup = BeautifulSoup(webContent, 'html.parser')

    # Extract text from the webpage
    text = soup.get_text()

    # clean text from url
    cleaned_text = clean_text(text)
    print("cleaned text from url")

    # Save the extracted text to a file
    with open(os.path.join('static', 'data', 'cleaned_data.txt'), 'w', encoding='utf-8') as file:
        file.writelines(cleaned_text)

    print("file saved")


def clean_text(text):
    # Remove references and non-textual elements
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
