from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils import nodes

class blogpost_node(nodes.Structural, nodes.Element):
    pass

class BlogPostDirective(Directive):
 
    # defines the parameter the directive expects
    # directives.unchanged means you get the raw value from RST
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'date': directives.unchanged,
                   'title': directives.unchanged,
                   'keywords': directives.unchanged,
                   'categories': directives.unchanged, }
    has_content = True
    add_index = True
    
    # import pdb; pdb.set_trace()
    def run(self):
        sett = self.state.document.settings
        language_code = sett.language_code
        env = self.state.document.settings.env
         
        # gives you access to the parameter stored
        # in the main configuration file (conf.py)
        config = env.config
         
        # gives you access to the options of the directive
        options = self.options
         
        # we create a section
        idb = nodes.make_id("blog-" + options["date"] + "-" + options["title"])
        section = nodes.section(ids=[idb])
         
        # we create a title and we add it to section
        section += nodes.title(options["title"])
         
        # we create the content of the blog post
        # because it contains any kind of RST
        # we parse parse it with function nested_parse
        par = nodes.paragraph()
        self.state.nested_parse(self.content, self.content_offset, par)
         
        # we create a blogpost and we add the section
        node = blogpost_node()
        node += section
        node += par
         
        # we return the result
        return [ node ]



