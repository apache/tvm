import os, platform, logging
from sitemaptree import SitemapTree
from datetime import datetime, timezone, timedelta


class DirToSitemap:
    def __init__(self, dir, html, root_url, home_page, change_freq, nsmap, priorities, time_zone, time_pattern):
        """
        init a sitemap three
        :param dir: file folder, absolute path
        :param html: whether url in sitemap contains ".html" suffix
        :param root_url: root url
        :param home_page: homepage file name
        :param change_freq:
        :param nsmap: xml namespaces
        """
        self.sitemap_tree = SitemapTree(namespace=nsmap, file="")
        self.dir_path = dir
        self.html = html
        self.root_url = root_url
        self.home_page = home_page
        self.change_freq = change_freq
        self.priorities = priorities
        self.tz = timezone(timedelta(hours=time_zone))
        self.tp = time_pattern

    def change_lastmod(self, node, t):
        """
        update lastmod
        """
        if t.find('+') != -1:  # %Y-%m-%dT%H:%M:%S+00:00
            if self.tp == '%Y-%m-%dT%H:%M:%S+00:00':
                node.find('lastmod', namespaces=node.nsmap).text = t
            elif self.tp == '%Y-%m-%d':
                node.find('lastmod', namespaces=node.nsmap).text = t[0:t.find('T')]
            else:
                logging.error('sitemap time pattern should be %Y-%m-%dT%H:%M:%S+00:00 or %Y-%m-%d')
        else:
            node.find('lastmod', namespaces=node.nsmap).text = t

    def path_to_url(self, rpath, html):
        """
        get file url by rpath
        :param rpath:relative path
        :return:
        """
        # homepage file
        if rpath == self.home_page:
            return self.root_url
        if (platform.system() == 'Windows'):
            rpath = '/'.join(rpath.split('\\'))
        # add ".html" suffix
        if html is True:
            if rpath[-5:] != ".html":
                rpath = rpath + ".html"
        else:
            if rpath[-5:] == ".html":
                rpath = rpath[0:-5]
        url = self.root_url + '/' + rpath
        return url

    def get_priority(self, rpath):
        """
        get priority by relative path
        :param rpath: relative path
        :return:
        """
        if rpath == self.home_page:
            return self.priorities[0]
        if (platform.system() == 'Windows'):
            rpath = '/'.join(rpath.split('\\'))
        depth = rpath.count('/')
        return self.priorities[depth + 1]

    def add_file(self, rpath):
        """
        add file node to home tree
        :param rpath: relative path
        :return:
        """
        url = self.path_to_url(rpath, self.html)
        priority = self.get_priority(rpath)
        # get time
        current_time_utc = datetime.utcnow().replace(tzinfo=self.tz)
        lastmod = datetime.strftime(current_time_utc, self.tp)
        cur_node = self.sitemap_tree.add_url(loc=url, lastmod=lastmod, changefreq=self.change_freq, priority=priority)
        if cur_node == None:
            logging.error("add file " + rpath + " failed.")

    def parse_dir(self, rpath=""):
        """
        parse dir to sitemap tree
        :param rpath: relative path to dir's absolute path
        :return:
        """
        # get absolute path
        apath = os.path.join(self.dir_path, rpath)
        files = os.listdir(apath)
        for file_name in files:
            temp_path = os.path.join(apath, file_name)
            if os.path.isfile(temp_path):
                if file_name[-5:] == '.html':
                    self.add_file(os.path.join(rpath, file_name))
            else:
                self.parse_dir(os.path.join(rpath, file_name))
        return self.sitemap_tree

    def save(self, file_name):
        """
        save the sitemap
        :param file_name: absolute path
        :return:
        """
        self.sitemap_tree.save(file_name)
