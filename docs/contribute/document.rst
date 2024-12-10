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

.. _doc_guide:

Documentation
=============

.. contents::
  :depth: 2
  :local:

TVM documentation loosely follows the `formal documentation style described by
Divio <https://documentation.divio.com>`_. This system has been chosen because
it is a "simple, comprehensive and nearly universally-applicable scheme. It is
proven in practice across a wide variety of fields and applications."

This document describes the organization of TVM documentation, and how to write
new documentation. See `docs/README.md <https://github.com/apache/tvm/tree/main/docs#build-locally>`_
for instructions on building the docs.

The Four Document Types
***********************

Introductory Tutorials
----------------------

These are step by step guides to introduce new users to a project. An
introductory tutorial is designed to get a user engaged with the software
without necessarily explaining why the software works the way it does. Those
explanations can be saved for other document types. An introductory tutorial
focuses on a successful first experience. These are the most important docs to
turning newcomers into new users and developers. A fully end-to-end
tutorial — from installing TVM and supporting ML software, to creating and
training a model, to compiling to different architectures — will give a new
user the opportunity to use TVM in the most efficient way possible. A tutorial
teaches a beginner something they need to know. This is in contrast with a
how-to, which is meant to be an answer to a question that a user with some
experience would ask.

Tutorials need to be repeatable and reliable, because the lack of success means
a user will look for other solutions.

How-to Guides
-------------

These are step by step guides on how to solve particular problems. The user can
ask meaningful questions, and the documents provide answers. An examples of
this type of document might be, "how do I compile an optimized model for ARM
architecture?" or "how do I compile and optimize a TensorFlow model?" These
documents should be open enough that a user could see how to apply it to a new
use case. Practical usability is more important than completeness. The title
should tell the user what problem the how-to is solving.

How are tutorials different from how-tos? A tutorial is oriented towards the
new developer, and focuses on successfully introducing them to the software and
community. A how-to, in contrast, focuses on accomplishing a specific task
within the context of basic understanding. A tutorial helps to on-board and
assumes no prior knowledge. A how-to assumes minimum knowledge, and is meant to
guide someone to accomplish a specific task.

Reference
---------

Reference documentation describes how the software is configured and operated.
APIs, key functions, commands, and interfaces are all candidates for reference
documentation. These are the technical manuals that let users build their own
interfaces and programs. They are information oriented, focused on lists and
descriptions. You can assume that the audience has a grasp on how the software
works and is looking for specific answers to specific questions. Ideally, the
reference documentation should have the same structure as the code base and be
generated automatically as much as possible.

Architecture Guides
-------------------

Architecture Guides are explanations are background material on a topic. These
documents help to illuminate and understand the application environment. Why
are things the way they are? What were the design decisions, what alternatives
were considered, what are the RFCs describing the existing system? This
includes academic papers and links to publications relevant to the software.
Within these documents you can explore contradictory and conflicting position,
and help the reader make sense of how and why the software was built the way it
is. It's not the place for how-tos and descriptions on how to accomplish tasks.
They instead focus on higher level concepts that help with the understanding of
the project. Generally these are written by the architects and developers of
the project, but can useful to help both users and developers to have a deeper
understanding of why the software works the way it does, and how to contribute
to it in ways that are consistent with the underlying design principles.

Special considerations for TVM
------------------------------

The TVM community has some special considerations that require deviation from
the simple docs style outlined by Divio. The first consideration is that there
is frequently overlap between the user and developer communities. Many projects
document the developer and user experience with separate systems, but it is
appropriate to consider both in this system, with differentiations where
appropriate. As a result the tutorials and how-tos will be divided between
"User Guides" that focus on the user experience, and "Developer Guides" that
focus on the developer experience.

The next consideration is that there are special topics within the TVM
community that benefit from additional attention. Special "Topic Guides" can be
created to index existing material, and provide context on how to navigate that
material most effectively.

To facilitate newcomers, a special "Getting Started" section with installation
instructions, a overview of why to use TVM, and other first-experience
documents will be produced.


Technical Details
*****************

We use the `Sphinx <http://sphinx-doc.org>`_ for the main documentation.
Sphinx supports both reStructuredText and markdown. When possible, we
encourage reStructuredText as it has richer features. Note that the
Python doc-string and tutorials allow you to embed reStructuredText syntax.

See
`docs/README.md <https://github.com/apache/tvm/tree/main/docs#build-locally>`_
for instructions on building the docs.


Python Reference Documentation
------------------------------

We use the `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ format to
document the function and classes. The following snippet gives an example
docstring. We always document all the public functions, when necessary,
provide an usage example of the features we support (as shown below).

.. code:: python

    def myfunction(arg1, arg2, arg3=3):
        """Briefly describe my function.

        Parameters
        ----------
        arg1 : Type1
            Description of arg1

        arg2 : Type2
            Description of arg2

        arg3 : Type3, optional
            Description of arg3

        Returns
        -------
        rv1 : RType1
            Description of return type one

        Examples
        --------
        .. code:: python

            # Example usage of myfunction
            x = myfunction(1, 2)
        """
        return rv1

Be careful to leave blank lines between sections of your documents. In the
above case, there has to be a blank line before ``Parameters``, ``Returns`` and
``Examples`` in order for the doc to be built correctly. To add a new function to
the docs, we need to add the `sphinx.autodoc
<http://www.sphinx-doc.org/en/master/ext/autodoc.html>`_ rules to
`docs/reference/api/python <https://github.com/apache/tvm/tree/main/docs/reference/api/python>`_).
You can refer to the existing files under this folder on how to add the
functions.

C++ Reference Documentation
---------------------------

We use the doxygen format to document c++ functions. The following snippet
shows an example of c++ docstring.

.. code:: c++

    /*!
     * \brief Description of my function
     * \param arg1 Description of arg1
     * \param arg2 Descroption of arg2
     * \returns describe return value
     */
    int myfunction(int arg1, int arg2) {
      // When necessary, also add comment to clarify internal logics
    }

Besides documenting function usages, we also highly recommend contributors to
add comments about code logics to improve readability.

Sphinx Gallery How-Tos
----------------------

We use `sphinx-gallery <https://sphinx-gallery.github.io/>`_ to build many
Python how-tos. You can find the source code under `gallery
<https://github.com/apache/tvm/tree/main/gallery>`_.
One thing that worth noting is that the comment blocks are written in
reStructuredText instead of markdown so be aware of the syntax.

The how-to code will run on our build server to generate the document page. So
we may have a restriction like not being able to access a remote Raspberry Pi,
in such case add a flag variable to the tutorial (e.g. ``use_rasp``) and allow
users to easily switch to the real device by changing one flag. Then use the
existing environment to demonstrate the usage.

If you add a new categorization of how-to, you will need to add references to
`conf.py <https://github.com/apache/tvm/tree/main/docs/conf.py>`_ and the
`how-to index <https://github.com/apache/tvm/tree/main/docs/how-to/index.rst>`_

Refer to Another Location in the Document
-----------------------------------------
Please use sphinx's ``:ref:`` markup to refer to another location in the same doc.

.. code-block:: rst

   .. _document-my-section-tag

   My Section
   ----------

   You can use :ref:`document-my-section-tag` to refer to My Section.

Documents with Images / Figures
-------------------------------
reStructuredText's `figure <https://docutils.sourceforge.io/docs/ref/rst/directives.html#figure>`_
and `image <https://docutils.sourceforge.io/docs/ref/rst/directives.html#image>`_
elements allow a document to include an image URL.

Image files created for TVM documentation should reside in the `<https://github.com/tlc-pack/web-data>`_
repository, while the `.rst` files *using* those images should reside in the main TVM repostitory
(`<https://github.com/apache/tvm>`_).

This will require two Github Pull Requests, one for the image files and another for the `.rst` files.
Discussion between the contributor and reviewers may be necessary to coordinate the review process.

*IMPORTANT NOTE:* When using two Pull Requests as described above, please merge the
Pull Request in `<https://github.com/tlc-pack/web-data>`_ *before* merging
the Pull Request in `<https://github.com/apache/tvm>`_.
This helps ensure that all URL links in TVM's online documentation are valid.
