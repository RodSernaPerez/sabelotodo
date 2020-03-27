from abc import ABC

import requests

from bs4 import BeautifulSoup


class Newspaper(ABC):
    def _get_page(self, entity):
        r = requests.get(self.URL.format(entity))
        return r.content

    def _parse_web(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup
    
    def find_links(self):
        pass
        mydivs = soup.findAll("div", {"class": "content search-results"})[0]


class ElConfidencial:
    URL = "https://www.elconfidencial.com/buscar/2-6-1-3/0/1/10/desc/{}}/"
