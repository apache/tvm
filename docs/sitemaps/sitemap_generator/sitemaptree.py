import logging
from lxml import etree


class SitemapTree:
    def __init__(self, namespace="", file=""):
        """
        urlset is a list of URLs that should appear in the sitemap
        :param namespace: html namespace
        :param file: sitemap file, format is xml, if not none, parse a sitemap tree from the file
        """
        if file == "":  # init a sitemaptree with only one node
            self.urlset = etree.Element('urlset')
            self.nsmap = namespace
            self.urlset.attrib['xmlns'] = self.nsmap
            self.etree = etree.ElementTree(self.urlset)
        else:  # parse sitemaptree for xml file
            self.etree = etree.parse(file, etree.XMLParser())
            self.urlset = self.etree.getroot()
            self.nsmap = self.urlset.nsmap

    def get_root(self):
        """
        get etree root node
        :return:
        """
        return self.urlset

    def get_node(self, url):
        """
        find node by its url
        :param url: url of the html
        :return: url node
        """
        cnodes = self.urlset.getchildren()
        for cnode in cnodes:
            loc_node = cnode.find('loc', namespaces=cnode.nsmap)
            if loc_node == None:
                logging.error("there should be a loc in url,url is {},cnode is {}".format(url, cnode))
                continue
            if url == loc_node.text:
                return cnode
        return None

    def add_url(self, **kwargs):
        """
        add a url to urlset
        :param loc: url
        :param lastmod: time
        :param changefreq: change frequency
        :param priority:
        :return: etree node
        """
        url = etree.Element('url')
        # loc
        loc = etree.Element('loc')
        if kwargs.get('loc') != None:
            loc.text = kwargs['loc']
        else:
            logging.error('the url is None')
            return None
        url.append(loc)

        lastmod = etree.Element('lastmod')
        if kwargs.get('lastmod') != None:
            lastmod.text = kwargs['lastmod']
        else:
            logging.error(url + 'does not have last modified time')
            return None
        url.append(lastmod)

        changefreq = etree.Element('changefreq')
        if kwargs.get('changefreq') != None:
            changefreq.text = kwargs['changefreq']
        else:
            changefreq.text = 'weekly'
        url.append(changefreq)

        priority = etree.Element('priority')
        if kwargs.get('priority') != None:
            priority.text = str(kwargs['priority'])
        else:
            priority.text = str(0.5)
        url.append(priority)

        self.urlset.append(url)
        return url

    @staticmethod
    def get_url(node):
        """
        get url from node
        """
        loc_node = node.find('loc', namespaces=node.nsmap)
        if loc_node is None:
            t = etree.tostring(node, pretty_print=1).decode("utf-8")
            logging.error("node \n {} does not include loc".format(t))
            return ""
        return loc_node.text

    def sort(self):
        """
        sort by url
        """
        urls = self.urlset
        urls[:] = sorted(self.urlset, key=self.get_url)

    def save(self, file_name):
        """
        savt sitemap to xml
        :param file_name:
        :return:
        """
        try:
            self.etree.write(file_name, pretty_print=True, xml_declaration=True, encoding="utf-8")
            logging.info('Sitemap saved in: {}'.format(file_name))
        except:
            logging.error("save " + file_name + " sitemap failed")
