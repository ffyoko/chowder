import re
import requests
import traceback
from urllib.parse import quote
import sys
import getopt
# from imp import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')


class crawler:
    url = u''
    urls = []
    o_urls = []
    html = ''
    total_pages = 5
    current_page = 0
    next_page_url = ''
    timeout = 60
    headersParameters = {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/6.1 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    }

    def __init__(self, keyword):
        self.url = u'https://www.baidu.com/baidu?wd=' + \
            quote(keyword)+'&tn=monline_dg&ie=utf-8'

    def set_timeout(self, time):
        try:
            self.timeout = int(time)
        except:
            pass

    def set_total_pages(self, num):
        try:
            self.total_pages = int(num)
        except:
            pass

    def set_current_url(self, url):
        self.url = url

    def switch_url(self):
        if self.next_page_url == '':
            sys.exit()
        else:
            self.set_current_url(self.next_page_url)

    def is_finish(self):
        if self.current_page >= self.total_pages:
            return True
        else:
            return False

    def get_html(self):
        r = requests.get(self.url, timeout=self.timeout,
                         headers=self.headersParameters)
        if r.status_code == 200:
            self.html = r.text
            self.current_page += 1
        else:
            self.html = u''
            print('[ERROR]', self.url, u'get此url返回的http状态码不是200')

    def get_urls(self):
        o_urls = re.findall(
            'href\=\"(http\:\/\/www\.baidu\.com\/link\?url\=.*?)\" class\=\"c\-showurl\"', self.html)
        o_urls = list(set(o_urls))
        self.o_urls = o_urls
        next = re.findall(
            ' href\=\"(\/s\?wd\=[\w\d\%\&\=\_\-]*?)\" class\=\"n\"', self.html)
        if len(next) > 0:
            self.next_page_url = 'https://www.baidu.com'+next[-1]
        else:
            self.next_page_url = ''

    def get_real(self, o_url):
        r = requests.get(o_url, allow_redirects=False)
        if r.status_code == 302:
            try:
                return r.headers['location']
            except:
                pass
        return o_url

    def transformation(self):
        self.urls = []
        for o_url in self.o_urls:
            self.urls.append(self.get_real(o_url))

    def print_urls(self):
        for url in self.urls:
            print(url)

    def print_o_urls(self):
        for url in self.o_urls:
            print(url)

    def run(self):
        while(not self.is_finish()):
            c.get_html()
            c.get_urls()
            c.transformation()
            c.print_urls()
            c.switch_url()


if __name__ == '__main__':
    help = 'baidu_crawler.py -k <keyword> [-t <timeout> -p <total pages>]'
    keyword = None
    timeout = None
    totalpages = None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hk:t:p:")
    except getopt.GetoptError:
        print(help)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help)
            sys.exit()
        elif opt in ("-k", "--keyword"):
            keyword = arg
        elif opt in ("-t", "--timeout"):
            timeout = arg
        elif opt in ("-p", "--totalpages"):
            totalpages = arg
    if keyword == None:
        print(help)
        sys.exit()

    c = crawler(keyword)
    if timeout != None:
        c.set_timeout(timeout)
    if totalpages != None:
        c.set_total_pages(totalpages)
    c.run()
