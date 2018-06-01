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
rules to the `docs/api/python <https://github.com/dmlc/tvm/tree/master/docs/api/python>`_).
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
You can find the source code under `tutorials <https://github.com/dmlc/tvm/tree/master/tutorials>`_ quite self explanatory.
One thing that worth noting is that the comment blocks are written in reStructuredText instead of markdown so be aware of the syntax.

The tutorial code will run on our build server to generate the document page.
So we may have a restriction like not being able to access a remote Raspberry Pi,
in such case add a flag variable to the tutorial (e.g. `use_rasp`) and allow users to easily switch to the real device by changing one flag.
Then use the existing environment to demonstrate the usage.
