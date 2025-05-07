
import requests
from bs4 import BeautifulSoup
from queue import Queue
base_url = 'https://stardewvalleywiki.com'
import spacy

nlp = spacy.load('en_core_web_md')
seen_links = set()


def get_page_links(page_url, tld):
    if not isinstance(page_url, str) or not page_url.startswith(tld):
        return []
    page = requests.get(page_url)
    content = page.content
    soup = BeautifulSoup(content, 'html.parser')
    refs = soup.find_all('a')
    sdv_pages = []
    # appreciate my handling here
    for ref in refs:
        href = ref.get('href')
        # if no link or we've seen the link before, continue
        if not href:
            continue
        elif href in seen_links:
            continue
        # if the link is an image continue
        elif href.endswith('png') or href.endswith('jpg') or href.endswith('jpeg') or href.endswith('bmp') or href.endswith('gif') \
                or ":" in href:
            continue
        # there are links to the same page's paragraphs, etc. so let's not use those
        elif href.startswith("/") and "#" not in href and "?" not in href:
            sdv_pages.append(f"{base_url}{href}")
        # don't want to go outside the stardew valley wiki
        elif not href.startswith(base_url):
            continue
        elif "#" in href:
            if href.split("#")[0] in seen_links:
                continue
            else:
                sdv_pages.append(href.split("#")[0])
        elif "?" in href:
            if href.split("?")[0] in seen_links:
                continue
            else:
                sdv_pages.append(href.split("?")[0])
        else:
            sdv_pages.append(href)
    return sdv_pages


def get_text(page_url):
    # get a page's contents
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



def load_file_contents(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        contents = f.readlines()
    return contents



def parse_wiki(filename, max_docs):
    tld = "https://stardewvalleywiki.com"
    mainurl = "https://stardewvalleywiki.com/Stardew_Valley_Wiki"
    main_page_links = get_page_links(mainurl, tld)
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
        pages = get_page_links(link, tld)
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
    max_articles = 1000
    filename = "StardewContentsLarge.txt"
    parse_wiki(filename, max_articles)


