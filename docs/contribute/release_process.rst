..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _release_process:

Release Process
===============

.. contents::
  :depth: 2
  :local:

The release manager role in TVM means you are responsible for a few different things:

- Preparing release notes
- Preparing your setup
- Preparing for release candidates

  - Cutting a release branch
  - Informing the community of timing
  - Making code changes in that branch with necessary version updates

- Running the voting process for a release

  - Creating release candidates
  - Calling votes and triaging issues

- Finalizing and posting a release:

  - Updating the TVM website
  - Finalizing release notes
  - Announcing the release


Prepare the Release Notes
-------------------------

Release note contains new features, improvement, bug fixes, known issues and deprecation, etc. TVM provides `monthly dev report <https://discuss.tvm.ai/search?q=TVM%20Monthly%20%23Announcement>`_ collects developing progress each month. It could be helpful to who writes the release notes.

It is recommended to open a Github issue to collect feedbacks for the release note draft before cutting the release branch. See the scripts in ``tests/scripts/release`` for some starting points.


Prepare the Release Candidate
-----------------------------

There may be some code changes necessary to the release branch before the release. Ensure all version numbers are up to date


Prepare the GPG Key
-------------------

You can skip this section if you have already uploaded your key.

After generating the gpg key, you need to upload your key to a public key server. Please refer to https://www.apache.org/dev/openpgp.html#generate-key for details.

If you want to do the release on another machine, you can transfer your gpg key to that machine via the :code:`gpg --export` and :code:`gpg --import` commands.

The last step is to update the KEYS file with your code signing key https://www.apache.org/dev/openpgp.html#export-public-key. Check in the changes to the TVM main branch, as well as ASF SVN,

.. code-block:: bash

	# the --depth=files will avoid checkout existing folders
	svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
	cd svn-tvm
	# edit KEYS file
	svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Update KEYS"
	# update downloads.apache.org (note that only PMC members can update the dist/release directory)
	svn rm --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/release/tvm/KEYS -m "Update KEYS"
	svn cp --username $ASF_USERNAME --password "$ASF_PASSWORD" https://dist.apache.org/repos/dist/dev/tvm/KEYS https://dist.apache.org/repos/dist/release/tvm/ -m "Update KEYS"


Cut a Release Candidate
-----------------------

To cut a release candidate, one needs to first cut a branch using selected version string, e.g.,

.. code-block:: bash

	git clone https://github.com/apache/tvm.git
	cd tvm/
	# Replace v0.6.0 with the relevant version
	git branch v0.6.0
	git push --set-upstream origin v0.6.0

(*Make sure the version numbers in the source code are correct.* Run :code:`python3 version.py` to update the version.)

Go to the GitHub repositories "releases" tab and click "Draft a new release",

- Provide the release tag in the form of “v1.0.0.rc0” where 0 means it’s the first release candidate
- Select the commit by clicking Target: branch > Recent commits > $commit_hash
- Copy and paste release note draft into the description box
- Select "This is a pre-release"
- Click "Publish release"

Notice that one can still apply changes to the BRANCH after the cut, while the TAG is fixed. If any change is required for this release, a new TAG has to be created.

Remove previous release candidate (if applied),

.. code-block:: bash

	git push --delete origin v0.6.0.rc1

Create source code artifacts,

.. code-block:: bash

	# Replace v0.6.0 with the relevant version
	git clone git@github.com:apache/tvm.git apache-tvm-src-v0.6.0
	cd apache-tvm-src-v0.6.0
	git checkout v0.6
	git submodule update --init --recursive
	git checkout v0.6.0.rc0
	rm -rf .DS_Store
	find . -name ".git*" -print0 | xargs -0 rm -rf
	cd ..
	brew install gnu-tar
	gtar -czvf apache-tvm-src-v0.6.0.rc0.tar.gz apache-tvm-src-v0.6.0

Use your GPG key to sign the created artifact. First make sure your GPG is set to use the correct private key,

.. code-block:: bash

	$ cat ~/.gnupg/gpg.conf
	default-key F42xxxxxxxxxxxxxxx

Create GPG signature as well as the hash of the file,

.. code-block:: bash

	gpg --armor --output apache-tvm-src-v0.6.0.rc0.tar.gz.asc --detach-sig apache-tvm-src-v0.6.0.rc0.tar.gz
	shasum -a 512 apache-tvm-src-v0.6.0.rc0.tar.gz > apache-tvm-src-v0.6.0.rc0.tar.gz.sha512


Update TVM Version on Main
--------------------------

After cutting a release candidate, make sure to update the version numbers throughout `main`. For example if we are
releasing `v0.10.0` we want to bump the version numbers throughout the codebase from `v0.10.dev0` to `v0.11.dev0`. An
example of how to do this can be found here: `https://github.com/apache/tvm/pull/12190 <https://github.com/apache/tvm/pull/12190>`_.

Upload the Release Candidate
----------------------------

Edit the release page on Github and upload the artifacts created by the previous steps.

The release manager also needs to upload the artifacts to ASF SVN,

.. code-block:: bash

	# the --depth=files will avoid checkout existing folders
	svn co --depth=files "https://dist.apache.org/repos/dist/dev/tvm" svn-tvm
	cd svn-tvm
	mkdir tvm-v0.6.0-rc0
	# copy files into it
	svn add tvm-0.6.0-rc0
	svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m "Add RC"


Call a Vote on the Release Candidate
------------------------------------

The first voting takes place on the Apache TVM developers list (dev@tvm.apache.org). To get more attention, one can create a github issue start with "[VOTE]" instead, it will be mirrored to dev@ automatically. Look at past voting threads to see how this proceeds. The email should follow this format.

- Provide the link to the draft of the release notes in the email
- Provide the link to the release candidate artifacts
- Make sure the email is in text format and the links are correct

For the dev@ vote, there must be at least 3 binding +1 votes and more +1 votes than -1 votes. Once the vote is done, you should also send out a summary email with the totals, with a subject that looks something like [VOTE][RESULT] ....

In ASF, votes are open "at least" 72hrs (3 days). If you don't get enough number of binding votes within that time, you cannot close the voting deadline. You need to extend it.

If the voting fails, the community needs to modified the release accordingly, create a new release candidate and re-run the voting process.


Post the Release
----------------

After the vote passes, to upload the binaries to Apache mirrors, you move the binaries from dev directory (this should be where they are voted) to release directory. This "moving" is the only way you can add stuff to the actual release directory. (Note: only PMC can move to release directory)

.. code-block:: bash

	export SVN_EDITOR=vim
	svn mkdir https://dist.apache.org/repos/dist/release/tvm
	svn mv https://dist.apache.org/repos/dist/dev/tvm/tvm-v0.6.0-rc2 https://dist.apache.org/repos/dist/release/tvm/tvm-v0.6.0

	# If you've added your signing key to the KEYS file, also update the release copy.
	svn co --depth=files "https://dist.apache.org/repos/dist/release/tvm" svn-tvm
	curl "https://dist.apache.org/repos/dist/dev/tvm/KEYS" > svn-tvm/KEYS
	(cd svn-tvm && svn ci --username $ASF_USERNAME --password "$ASF_PASSWORD" -m"Update KEYS")

Remember to create a new release TAG (v0.6.0 in this case) on Github and remove the pre-release candidate TAG.

 .. code-block:: bash

    git push --delete origin v0.6.0.rc2


Update the TVM Website
----------------------

The website repository is located at `https://github.com/apache/tvm-site <https://github.com/apache/tvm-site>`_. Modify the download page to include the release artifacts as well as the GPG signature and SHA hash. Since TVM's docs are continually updated, upload a fixed version of the release docs. If CI has deleted the docs from the release by the time you go to update the website, you can restart the CI build for the release branch on Jenkins. See the example code below for a starting point

.. code-block:: bash

	git clone https://github.com/apache/tvm-site.git
	pushd tvm-site
	git checkout asf-site
	pushd docs

	# make release docs directory
	mkdir v0.9.0
	pushd v0.9.0

	# download the release docs from CI
	# find this URL by inspecting the CI logs for the most recent build of the release branch
	curl -LO https://tvm-jenkins-artifacts-prod.s3.us-west-2.amazonaws.com/tvm/v0.9.0/1/docs/docs.tgz
	tar xf docs.tgz
	rm docs.tgz

	# add the docs and push
	git add .
	git commit -m "Add v0.9.0 docs"
	git push


Afterwards, modify the `downloads page <https://tvm.apache.org/download>`_ to support the latest release. An example of how to do this is `here <https://github.com/apache/tvm-site/pull/38>`_.

Post the Announcement
---------------------

Send out an announcement email to announce@apache.org, and dev@tvm.apache.org. The announcement should include the link to release note and download page.
