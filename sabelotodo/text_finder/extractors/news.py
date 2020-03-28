from abc import ABC, abstractmethod
from typing import List

import requests
from bs4 import BeautifulSoup
from newspaper import Article

from sabelotodo.text_finder.extractors.extractor import Extractor
from sabelotodo.text_piece import TextPiece


class Newspaper(Extractor, ABC):
    URL = ""
    LANGUAGE = ""

    def _get_page(self, entity) -> str:
        r = requests.get(self.URL.format(entity))
        return r.content

    def _parse_web(self, html) -> BeautifulSoup:
        soup = BeautifulSoup(html, 'html.parser')
        return soup

    @abstractmethod
    def _find_links(self, parsed_web) -> List[str]:
        pass

    def extract_topic(self, entity) -> List[TextPiece]:
        page = self._get_page(entity)
        parsed_web = self._parse_web(page)
        links = self._find_links(parsed_web)
        articles = [self._download_article(link) for link in links]
        return articles

    def _download_article(self, link) -> TextPiece:
        article = Article(link)
        article.download()
        article.parse()
        return TextPiece(article.title, article.text)


class ElConfidencial(Newspaper):
    URL = "https://www.elconfidencial.com/buscar/2-6-1-3/0/1/10/desc/{}/"
    LANGUAGE = "es"

    _NOT_VALID_LINKS = ["https://www.elconfidencial.com/buscar/",
                        "https://www.elconfidencial.com/empresas/"]

    def _find_links(self, soup_web) -> List[str]:
        div_resultados = soup_web.findAll("div", {"class": "content search-results"})[0]
        soup_resultados = BeautifulSoup(str(div_resultados), 'html.parser')
        links = list({a["href"] for a in soup_resultados.find_all('a', href=True)})
        links = self._filter_links(links)
        return links

    def _filter_links(self, links) -> List[str]:
        """Some links that do not belong to news could appear. This method tries to filter them."""
        links = [link for link in links if self._is_valid_link(link)]
        return links

    def _is_valid_link(self, link) -> bool:
        return not any([x in link for x in self._NOT_VALID_LINKS]) and \
               len(link)  # Not Empty
