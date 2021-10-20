import os
import time
import argparse
import logging
from dirtositemap import DirToSitemap
from config import *
from sitemaptree import SitemapTree


def cmp_file(f1, f2):
    st1 = os.stat(f1)
    st2 = os.stat(f2)

    # compare file size
    if st1.st_size != st2.st_size:
        return False

    bufsize = 8 * 1024
    with open(f1, 'rb') as fp1, open(f2, 'rb') as fp2:
        while True:
            b1 = fp1.read(bufsize)
            b2 = fp2.read(bufsize)
            if b1 != b2:
                return False
            if not b1:
                logging.info("{} and {} isn't change".format(f1, f2))
                return True


def parse_dir(dir, cur_path=""):
    """
    get html file and path
    :param dir: dir path, absolute path
    :return: dict{rpath:filename}
    """
    result = {}
    apath = os.path.join(dir, cur_path)
    files = os.listdir(apath)
    for file_name in files:
        temp_path = os.path.join(apath, file_name)
        rpath = os.path.join(cur_path, file_name)
        if os.path.isfile(temp_path):
            if file_name[-5:] == '.html':
                result[rpath] = file_name
        else:
            result.update(parse_dir(dir, rpath))
    return result


def compare(old_dir, new_dir, old_sitemap, html):
    """

    :param old_dir: absolute path
    :param new_dir: absolute path
    :param old_sitemap: html_old's sitemap
    :return:
    """
    # sitemaptree for dir html
    sitemap = DirToSitemap(dir=new_dir, html=html, root_url=ROOTURL, home_page=HOMEPAGE,
                           change_freq=CHANGEFREQ_PATTERNS[3], nsmap=XMLNS, priorities=PRIORITIES, time_zone=TIMEZONE,
                           time_pattern=LASTMODFORMAT)
    pt = sitemap.parse_dir("")

    # if old_sitemap is None, or old_dir is None
    if old_sitemap == None or old_dir == None:
        return pt
    if os.path.exists(old_sitemap) == False:
        logging.error("there is no old sitemap in {}".format(old_sitemap))
        return pt
    if os.path.exists(old_dir) == False:
        logging.error("there is no old dir in {}".format(old_dir))
        return pt

    # sitemaptree for dir html_old
    pt_old = SitemapTree(file=old_sitemap)
    path_file_dic = parse_dir(old_dir)
    for rpath, file in path_file_dic.items():
        old_apath, new_apath = os.path.join(old_dir, rpath), os.path.join(new_dir, rpath)
        if os.path.exists(new_apath) and os.path.exists(old_apath):
            if cmp_file(old_apath, new_apath) == True:  # update lastmod
                url_html = sitemap.path_to_url(rpath, True)
                url_nhtml = sitemap.path_to_url(rpath, False)
                if sitemap.html == True:
                    new_node = pt.get_node(url_html)
                else:
                    new_node = pt.get_node(url_nhtml)

                if new_node == None:
                    logging.error(
                        "the node in new sitemap should not be none, path is {},url is {}".format(rpath, url_html))
                old_node = pt_old.get_node(url_html)
                if old_node == None:  # maybe some url in old sitemap are not ended with ".html"
                    old_node = pt_old.get_node(url_nhtml)

                if old_node == None:  # did not find the node in old sitemap
                    logging.error("no site map for file in {}".format(old_apath))
                    continue
                logging.info("change file {} lastmod".format(rpath))
                old_lastmod = old_node.find('lastmod', namespaces=old_node.nsmap).text
                sitemap.change_lastmod(new_node, old_lastmod)
    return pt


# if __name__ == "__main__":
logging.basicConfig(level=logging.ERROR,
                    format=LOGGINTFORMAT,
                    )
# generate sitemap by comparing html dir and old html dir
parser = argparse.ArgumentParser()
parser.add_argument('--ndir', help="new dir absolute path")
parser.add_argument('--odir', help="old dir absolute path")
parser.add_argument('--ositemap', help="old sitemap absolute path")
parser.add_argument('--sitemap', help="new sitemap absoluth path", default="")
parser.add_argument('--html', action='store_false', help="contains .html suffix, default true")

args = parser.parse_args()

pt = compare(args.odir,
             args.ndir,
             args.ositemap,
             args.html)
pt.sort()
pt.save(os.path.abspath(args.sitemap))
