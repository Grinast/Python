# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:36:41 2018

@author: Cristina
"""

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

#Task 1:
#download webpage contents (raw HTML)

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    
    The closing() function ensures that any network resources are freed when 
    they go out of scope in that with block. Using closing() like that is good 
    practice and helps to prevent fatal errors and network timeouts.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        log_error("Error during requests to {0} : {1}".format(url,str(e)))
        return None
    
def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers["Content-Type"].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find("html")> -1)
    
def log_error(e):
    """
    It is always a good idea to log errors. 
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


#Task 2
#Decode HTML and extract useful information

#For this purpose, you will be using BeautifulSoup. 
#The BeautifulSoup constructor parses raw HTML strings and produces an object 
#that mirrors the HTML document’s structure. 



url = "http://minimini.jp/list/line/tokyo/keiodentetsukeiosen/naganuma/?lnkdiv=8"

raw_html = simple_get(url)

if raw_html is not None:
    html = BeautifulSoup(raw_html, "html.parser")
else:
    # Raise an exception if we failed to get any data from the url
    raise Exception("Error retrieving contents at {}".format(url))


def find_link_text(html):
    """
    Given html code returns the text in each link as a list
    <a href="..."> text </a>
    """
    link_text = []
    #find all links
    links = html.find_all("a")
    for link in links:
        link_text.append(link.text)
    return link_text

def find_link_text_clean(html):
    """
    Given html code returns the text in each link as a list.
    Ignores texts which are empty strings
    <a href="..."> text </a>
    """
    link_text = []
    #find all links
    links = html.find_all("a")
    for link in links:
        if link.text != "":
            link_text.append(link.text)
    return link_text
      
#get houses names
def get_names(html):
    """
    Given html code returns the name of each property as a list.
    <h3><a href="..."> house_name </a></h3>
    """
    names = []
    links = html.select("h3 a")
    for link in links:
        names.append(link.text)
    return names

#get locations
def get_locations(html):
    """
    Given html code returns the location of each property as a list.
    <table class="tateya_table">
    <tr>
    <th>所在地</th><td>location</td>
    <th>建築年月</th><td>...</td>
    </tr>
    <tr>
    <th>最寄駅</th><td>...</td>
    </tr>
    </table>
    """
    locations = []
    #find all tables of the class tateya_table
    tables = html.select(".tateya_table")
    for table in tables:
        #for each table select the first <td> element
        loc = table.select("td:nth-of-type(1)")
        #parse the text of the list (the list contains only one element)
        locations.append(loc[0].text)
    return locations

#get prices
    

#get room setup (2K, 3LD, etc.)
    




    
def get_names():
    """
    Downloads the page where the list of mathematicians is found
    and returns a list of strings, one per mathematician
    """
    url = 'http://www.fabpedigree.com/james/mathmen.htm'
    response = simple_get(url)

    if response is not None:
        html = BeautifulSoup(response, 'html.parser')
        names = set()
        for li in html.select('li'):
            for name in li.text.split('\n'):
                if len(name) > 0:
                    names.add(name.strip())
        return list(names)

    # Raise an exception if we failed to get any data from the url
    raise Exception('Error retrieving contents at {}'.format(url))
    
    