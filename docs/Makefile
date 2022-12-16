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

# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -m sphinx
PAPER         =
BUILDDIR      = _build
STAGINGDIR    = _staging

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(PWD)/$(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest coverage gettext staging

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  applehelp  to make an Apple Help Book"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  latexpdfja to make LaTeX files and run them through platex/dvipdfmx"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  xml        to make Docutils-native XML files"
	@echo "  pseudoxml  to make pseudoxml-XML files for display purposes"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage   to run coverage check of the documentation (if enabled)"

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(STAGINGDIR)

	# TODO(Lunderberg): Remove these lines once the CI steps have
	# propagated.

	# Remove folders that have since been relocated into
	# $(STAGINGDIR).  This allows `task_python_docs.sh` to
	# run, even if a commit that predates $(STAGINGDIR) was
	# previously run on that node.
	rm -rf gen_modules
	rm -rf user_tutorials
	rm -rf tutorials
	rm -rf vta/tutorials

staging:
	# Prepare the staging directory.  Sphinx gallery automatically
	# writes new .rst files into the current directory.  This can
	# cause issues when switching branches.  By sequestering the
	# auto-generated files into the staging directory, they can be
	# removed without knowing the exact directory.

	mkdir -p $(STAGINGDIR)

	# Remove any symlinks that currently exist
	find $(STAGINGDIR) -type l -exec rm {} \;

	# Reproduce the directory structure
	find . \
	      -path ./$(BUILDDIR) -prune -o -path ./$(STAGINGDIR) -prune -o \
	      -name "*.rst"  \
	      -printf "$(STAGINGDIR)/%h\n" \
              | sort | uniq | xargs mkdir -p

	# Symlink all .rst files into the staging directory
	find . \
	      -path ./$(BUILDDIR) -prune -o -path ./$(STAGINGDIR) -prune -o \
	      -name "*.rst"  \
	      -exec ln -s $(PWD)/{} $(STAGINGDIR)/{} \;

	ln -s $(PWD)/conf.py $(STAGINGDIR)/conf.py
	ln -s $(PWD)/_static $(STAGINGDIR)/_static


html: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/rabit.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/rabit.qhc"

applehelp: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b applehelp $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/applehelp
	@echo
	@echo "Build finished. The help book is in $(BUILDDIR)/applehelp."
	@echo "N.B. You won't be able to view it unless you put it in" \
	      "~/Library/Documentation/Help or install it in your application" \
	      "bundle."

devhelp: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/rabit"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/rabit"
	@echo "# devhelp"

epub: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

latexpdfja: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/latex
	@echo "Running LaTeX files through platex and dvipdfmx..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf-ja
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C $(BUILDDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(BUILDDIR)/texinfo."

gettext: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(PWD)/$(BUILDDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

changes: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

coverage: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/coverage
	@echo "Testing of coverage in the sources finished, look at the " \
	      "results in $(BUILDDIR)/coverage/python.txt."

xml: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b xml $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/xml
	@echo
	@echo "Build finished. The XML files are in $(BUILDDIR)/xml."

pseudoxml: staging
	cd $(STAGINGDIR) && $(SPHINXBUILD) -b pseudoxml $(ALLSPHINXOPTS) $(PWD)/$(BUILDDIR)/pseudoxml
	@echo
	@echo "Build finished. The pseudo-XML files are in $(BUILDDIR)/pseudoxml."
