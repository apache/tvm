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

Write Document and Tutorials
============================

We use the `Sphinx <http://sphinx-doc.org>`_ for the main documentation.
Sphinx support both the reStructuredText and markdown.
When possible, we encourage to use reStructuredText as it has richer features.
Note that the python doc-string and tutorials allow you to embed reStructuredText syntax.


Document Python
---------------
We use `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_
format to document the function and classes.
The following snippet gives an example docstring.
We always document all the public functions,
when necessary, provide an usage example of the features we support(as shown below).

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

Be careful to leave blank lines between sections of your documents.
In the above case, there has to be a blank line before `Parameters`, `Returns` and `Examples`
in order for the doc to be built correctly. To add a new function to the doc,
we need to add the `sphinx.autodoc <http://www.sphinx-doc.org/en/master/ext/autodoc.html>`_
rules to the `docs/api/python <https://github.com/apache/incubator-tvm/tree/main/docs/api/python>`_).
You can refer to the existing files under this folder on how to add the functions.


Document C++
------------
We use the doxgen format to document c++ functions.
The following snippet shows an example of c++ docstring.

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

Besides documenting function usages, we also highly recommend contributors
to add comments about code logics to improve readability.


Write Tutorials
---------------
We use the `sphinx-gallery <https://sphinx-gallery.github.io/>`_ to build python tutorials.
You can find the source code under `tutorials <https://github.com/apache/incubator-tvm/tree/main/tutorials>`_ quite self explanatory.
One thing that worth noting is that the comment blocks are written in reStructuredText instead of markdown so be aware of the syntax.

The tutorial code will run on our build server to generate the document page.
So we may have a restriction like not being able to access a remote Raspberry Pi,
in such case add a flag variable to the tutorial (e.g. `use_rasp`) and allow users to easily switch to the real device by changing one flag.
Then use the existing environment to demonstrate the usage.


Refer to Another Location in the Document
-----------------------------------------
Please use sphinx's `:ref:` markup to refer to another location in the same doc.

.. code-block:: rst

   .. _document-my-section-tag

   My Section
   ----------

   You can use :ref:`document-my-section-tag` to refer to My Section.
