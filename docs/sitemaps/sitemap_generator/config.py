# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# root url
ROOTURL = "https://example.com"

# homepage"s filename
HOMEPAGE = "index.html"

# Text encodings
ENC_UTF8 = "UTF-8"

# General Sitemap tags
GENERAL_SITEMAP_TAGS = ["loc", "changefreq", "priority", "lastmod"]

# Match patterns for changefreq attributes
CHANGEFREQ_PATTERNS = ["always", "hourly", "daily", "weekly", "monthly", "yearly", "never"]

# priorities by file depth in folder
PRIORITIES = ["1.0", "0.8", "0.64", "0.51", "0.41", "0.33", "0.26", "0.21"]

# sitemap namespace
XMLNS = "http://www.sitemaps.org/schemas/sitemap/0.9"


# lastmod format can be "%Y-%m-%dT%H:%M:%S+08:00" or "%Y-%m-%d"
LASTMODFORMAT = "%Y-%m-%d"

# log format
LOGGINTFORMAT = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"

# time zone, Beijing
TIMEZONE = 8
