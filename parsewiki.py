
import requests
from bs4 import BeautifulSoup
from queue import Queue
base_url = 'https://stardewvalleywiki.com'
import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = spacy.load('en_core_web_sm')
# for paragraph in paragraphs:
#     print(paragraph.text)
seen_links = set()


def get_page_links(page_url):
    if not isinstance(page_url, str) or not page_url.startswith(tld):
        return []
    page = requests.get(page_url)
    content = page.content
    soup = BeautifulSoup(content, 'html.parser')
    refs = soup.find_all('a')
    sdv_pages = []
    for ref in refs:
        href = ref.get('href')
        if not href:
            continue
        elif href in seen_links:
            continue
        elif href.endswith('png') or href.endswith('jpg') or href.endswith('jpeg') or href.endswith('bmp') or href.endswith('gif') \
                or ":" in href:
            continue
        elif href.startswith("/") and "#" not in href and "?" not in href:
            # seen_links.add(f"{base_url}{href}")
            sdv_pages.append(f"{base_url}{href}")
        elif not href.startswith(base_url):
            continue
        elif "#" in href:
            if href.split("#")[0] in seen_links:
                continue
            else:
                # seen_links.add(href.split("#")[0])
                sdv_pages.append(href.split("#")[0])
        elif "?" in href:
            if href.split("?")[0] in seen_links:
                continue
            else:
                # seen_links.add(href.split("?")[0])
                sdv_pages.append(href.split("?")[0])

        else:
            # seen_links.add(href)
            sdv_pages.append(href)
    return sdv_pages



def get_main_page_links(page_url):
    links = []
    for page in get_page_links(page_url)[1:]:
        if not page.has_key('href') or page['href'] is None:
            continue
        if page['href'].startswith('/'):
            links.append(f"{tld}{page['href']}")
            seen_links.add(f"{tld}{page['href']}")
        elif not page['href'].startswith('https://stardewvalleywiki.com'):
            continue

    return links


def get_text(page_url):
    page = requests.get(page_url)

    content = page.content
    soup = BeautifulSoup(content, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join([paragraph.text for paragraph in paragraphs])


def get_unseen_pages(candidates):
    unseen_pages = []
    for candidate in candidates:
        if candidate not in seen_links:
            unseen_pages.append(candidate)
    seen_links.update(unseen_pages)
    return unseen_pages


def get_page_tokens(page_contents):
    doc = nlp.tokenizer(page_contents)
    return doc


def save_to_file(filename: str, line: str):
    with open(filename, 'a+') as f:
        line += "\n"
        f.write(line)


def remove_newlines(line: str):
    return line.replace('\n', '').strip()


def retrieve_and_save_contents():
    tld = "https://stardewvalleywiki.com"
    mainurl = "https://stardewvalleywiki.com/Stardew_Valley_Wiki"
    filename = "StardewContents.txt"
    main_page_links = get_page_links(tld)
    page_queue = Queue()
    for page_url in main_page_links:
        page_queue.put(page_url)

    print(main_page_links)
    count = 0
    # while not page_queue.empty():
    corpus = []
    while not page_queue.empty():
        link = page_queue.get()
        seen_links.add(link)
        # print(new_links)
        # term_term_matrix = pd.DataFrame()
        text = get_text(link)  # long string
        text = remove_newlines(text)

        save_to_file(filename, text)
        # doc = get_page_tokens(text)  # spacy doc
        # corpus.extend([token.text for token in doc])   # just words
        pages = get_page_links(link)
        unseen_pages = get_unseen_pages(pages)
        for unseen_page in unseen_pages:
            page_queue.put(unseen_page)
        seen_links.update(pages)
        count += 1
        if count % 10 == 0:
            print(".", end='')
            if count % 100 == 0:
                print(f"Queue Size {page_queue.qsize()}")


def load_file_contents(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        contents = f.readlines()
    return contents



def parse_wiki(filename, max_docs):
    tld = "https://stardewvalleywiki.com"
    mainurl = "https://stardewvalleywiki.com/Stardew_Valley_Wiki"
    main_page_links = get_page_links(tld)
    page_queue = Queue()
    for page_url in main_page_links:
        page_queue.put(page_url)

    count = 0
    corpus = []
    while not page_queue.empty() and count < max_docs:
        link = page_queue.get()
        seen_links.add(link)
        text = get_text(link)  # long string
        text = remove_newlines(text)

        save_to_file(filename, text)
        pages = get_page_links(link)
        unseen_pages = get_unseen_pages(pages)
        for unseen_page in unseen_pages:
            page_queue.put(unseen_page)
        seen_links.update(pages)
        count += 1
        if count % 10 == 0:
            print(".", end='')
            if count % 100 == 0:
                print(f"Queue Size {page_queue.qsize()}")


if __name__ == "__main__":

    # model = gensim.models.Word2Vec([corpus], vector_size=100, window=5, min_count=1, workers=4)
    # print(model.wv['Stardew'])


    # cv = CountVectorizer(ngram_range=(1,3), stop_words="english")
    # counts = cv.fit_transform(corpus).toarray()
    # pca = PCA(n_components=10)
    # pca.fit(counts)
    # t = pca.transform(counts)
    # print(counts)




        # pages = get_page_links(link)
        # unseen_pages = get_unseen_pages(pages)
        # for unseen_page in unseen_pages:
        #     page_queue.put(unseen_page)
        # seen_links.update(pages)
        # count += 1
        # if count % 50 == 0:
        #     print(f"Processed {count} pages, queue size: {page_queue.qsize()}")

    print(count)