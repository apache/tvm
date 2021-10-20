# root url
ROOTURL = "https://tvmchinese.github.io"

# homepage's filename
HOMEPAGE = "index.html"

# Text encodings
ENC_UTF8 = 'UTF-8'

# General Sitemap tags
GENERAL_SITEMAP_TAGS = [
    'loc', 'changefreq', 'priority', 'lastmod'
]

# Match patterns for changefreq attributes
CHANGEFREQ_PATTERNS = [
    'always', 'hourly', 'daily', 'weekly', 'monthly', 'yearly', 'never'
]

# priorities by file depth in folder
PRIORITIES = [
    "1.0", "0.8", "0.64", "0.51", "0.41", "0.33", "0.26", "0.21"
]

# sitemap namespace
XMLNS = "http://www.sitemaps.org/schemas/sitemap/0.9"


# lastmod format can be '%Y-%m-%dT%H:%M:%S+08:00' or '%Y-%m-%d'
LASTMODFORMAT = '%Y-%m-%d'

# log format
LOGGINTFORMAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

# time zone, Beijing
TIMEZONE = 8
