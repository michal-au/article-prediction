from typing import Iterable, List


class Tree:
    def __init__(self, val, parent=None, children=[], order_nb=None):
        self.val = val  # type: str
        self.parent = parent  # type: Tree
        self.children = children  # type: List[Tree]
        self.order_nb = order_nb  # type: int  # zero-based

    def __repr__(self):
        stack = []
        depth = 1
        string = self.val + "\n"
        stack.extend([(child, depth) for child in reversed(self.children)])
        while len(stack) > 0:
            (node, depth) = stack.pop()
            string += "  "*depth + node.val + "\n"
            stack.extend([(child, depth+1)
                          for child in reversed(node.children)])
        return string[:string.rfind('\n')]  # remove last (empty) line

    def insert_article(self, article_form):
        """
        This breaks the tree structure... the tree can be printed but nothing more
        """
        r = self.get_root()
        if self.get_leftmost_child().order_nb == 0:
            article_form = article_form.capitalize()
        if all([w.upper() == w for w in r.get_words()]):
            article_form = article_form.upper()
        T = Tree.from_string('(ROOT(DT {}))'.format(article_form))
        self.children.insert(0, T.children[0])

    def print_sentence(self):
        "`', 's,"
        s = ''
        for n in self:
            if not n.is_leaf():
                continue
            token = n.get_word_form()
            if token == '-LRB-':
                token = '('
            if token == '-RRB-':
                token = ')'
            if n.get_label() in (',', '.', '!', '?', ';', ':', "'s"):
                s += token
            else:
                s += ' ' + token
        return s

    def get_right_siblings(self):
        if self.is_root():
            return None
        sibs = self.parent.children
        idx = [i for i, node in enumerate(sibs) if node is self][0]
        #if idx+1 >= len(sibs):
        #    return None
        return sibs[idx+1:]

    def get_left_siblings(self):
        if self.is_root():
            return None
        sibs = self.parent.children
        idx = [i for i, node in enumerate(sibs) if node is self][0]
        return sibs[:idx]

    def getChildren(self):
        return self.children

    def get_label(self):
        # type: () -> str
        return self.val.split('~')[0]

    def get_head(self):
        # TODO: test it
        idx = int(self.val.split('~')[3]) - 1
        chldrn = [ch for ch in self.getChildren() if ch.get_label() != 'PUNC']
        return chldrn[idx]

    """
    def get_head_collins_all(self):
        chldrn = [ch for ch in self.getChildren()]
        if self.get_label() in ('NP', 'NPB'):
            return self.get_head_collins()
        for ch in chldrn:
            if ch.get_label() in ('NNS', 'QP', 'NN', '$', 'ADVP', 'JJ', 'VBN', 'VBG', 'ADJP', 'JJR', 'NP', 'JJS', 'DT', 'FW', 'RBR', 'RBS', 'SBAR', 'RB'):
                return ch
        for ch in reversed(chldrn):
            if ch.get_label() in ('RB', 'RBR', 'RBS', 'FW', 'ADVP', 'TO', 'CD', 'JJR', 'JJ', 'IN', 'NP', 'JJS', 'NN'):
                return ch

        if self.get_label() == 'ADJP':
    """

    def get_head_collins(self):
        # type: () -> Tree
        """
        head finding rules according to the collins thesis:
        Michael Collins. 1999. Head-Driven Statistical Models for Natural Language Parsing, Ph.D. Thesis, University of Pennsylvania, Philadelphia, PA.
        pp.238-9
        """
        chldrn = [ch for ch in self.getChildren()]
        for ch in reversed(chldrn):
            if ch.get_label() in ['NN', 'NNP', 'NNPS', 'NNS', 'NX', 'JJR']:
                return ch
        for ch in chldrn:
            if ch.get_label() == 'NP':
                return ch
        for ch in reversed(chldrn):
            if ch.get_label() in ['$', 'ADJP', 'PRN']:
                return ch
        for ch in reversed(chldrn):
            if ch.get_label() == 'CD':
                return ch
        for ch in reversed(chldrn):
            if ch.get_label() in ['JJ', 'JJS', 'RB', 'QP']:
                return ch
        if len(chldrn) > 0:
            return chldrn[-1]

    def getPosTags(self):
        # type: () -> List[str]
        return [n.val for n in self if n.is_leaf()]

    def get_word_form(self):
        # type: () -> str
        if self.is_leaf:
            return self.children[0].val
        else:
            raise NameError("Did not get a leaf node!: {}".format(self.val))

    def get_words(self):
        # type: () -> List[str]
        return [n.children[0].val for n in self if n.is_leaf()]

    def get_word_tag_pairs(self):
        return [(n.children[0].val, n.val) for n in self if n.is_leaf()]

    def removeNodeByValue(self, value):
        #TODO: smazat a pouzit misto toho delete_nodes_by_value
        for n in self:
            if n.val == value:
                for i, sib in enumerate(n.parent.children):
                    if sib is n:
                        del n.parent.children[i]
        return self

    def is_leaf(self):
        return len(self.children) == 1 and self.children[0].children == []

    def is_root(self):
        return self.parent is None

    def get_root(self):
        # type: () -> Tree
        root = self
        while not root.is_root():
            root = root.parent
        return root

    def get_leftmost_child(self):
        # type: () -> Tree
        leftmost_child = self
        while not leftmost_child.is_leaf():
            leftmost_child = leftmost_child.children[0]
        return leftmost_child

    def get_rightmost_child(self):
        # type: () -> Tree
        rightmost_child = self
        while not rightmost_child.is_leaf():
            rightmost_child = rightmost_child.children[-1]
        return rightmost_child

    def get_main_verb(self):
        if self.get_label() != 'S':
            raise NameError('Main verb can be obtained only for "S" nodes.')

        # TODO: tohle muze fungovat jen s collins-parserem, takze dobry jen pro tuning
        # TODO: lemmatize
        return self.val.split('~')[1]
        #vp = [ch for ch in self.children if ch.get_label() == 'VP']
        #if len(vp) > 1:
        #    print ">>>>>>>>>", self
        #    #print 'VP', vp

    def delete_node(self):
        if self.is_root():
            return False
        for i, sib in enumerate(self.parent.children):
            if sib is self:
                del self.parent.children[i]
                return True
        return False

    def delete_nodes_by_value(self, value):
        for n in self:
            if n.val == value:
                if n.is_root():
                    raise NameError('trying to delete the root: {}'.format(self.to_string()))
                node = n.parent
                n.delete_node()
                while len(node.children) == 0:
                    u = node
                    if u.is_root():
                        raise NameError('trying to delete the root: {}'.format(self.to_string()))
                    node = node.parent
                    u.delete_node()

    def to_string(self, string=""):
        """converts the tree to the inline bracketed representation"""
        string += ' ({}'.format(self.val)
        if self.is_leaf():
            string += ' {})'.format(self.children[0].val)
        else:
            for ch in self.children:
                string = ch.to_string(string=string)
            string += ')'
        return string

    def print_highlighted(self):
        # type: () -> str
        """
        Returns the sentence with the self node highlighted (e.g. "This is [ a nice example ] . ")
        """
        root = self.get_root()
        words = root.get_words()
        l_idx, r_idx = self.get_leftmost_child().order_nb, self.get_rightmost_child().order_nb
        words = words[:l_idx] + ['['] + words[l_idx:r_idx+1]+ [']'] + words[r_idx+1:]
        return ' '.join(words)

    @classmethod
    def from_file(cls, fPath):
        """
        Constructs trees by parsing the given file. Used to process the original PTB data, where trees span multiple lines
        """
        with open(fPath) as f:
            s = ' '.join(line for line in f).replace('\n', ' ')
        s = s.replace('(', ' ( ')
        s = s.replace(')', ' ) ')
        nodes = s.split()

        trees = []
        # stack holds the ancestors of the given node
        stack = [Tree(nodes[2], None, [])]
        idx = 3  # position within the nodes list
        while idx < len(nodes):
            n = nodes[idx]
            if n == '(':
                chlds = stack[-1].children  # children of the parent node
                if nodes[idx+2] == '(':
                    # internal node:
                    newN = Tree(nodes[idx+1], stack[-1], [])
                    chlds.append(newN)
                    stack.append(newN)
                    idx += 2
                else:
                    # leaf and its POS tag:
                    newN = Tree(nodes[idx+1], stack[-1], [Tree(nodes[idx+2])])
                    newN.children[0].parent = newN
                    chlds.append(newN)
                    idx += 4
            elif n == ')':
                if len(stack) == 1:
                    trees.append(cls(stack[0].val, None, stack[0].children))
                    stack.pop()
                    if idx+5 < len(nodes):
                        stack.append(Tree(nodes[idx+4], None, []))
                        idx += 5
                        continue
                    else:
                        break
                else:
                    stack.pop()
                    idx += 1
            else:
                raise NameError("ill-formed tree")

        return trees

    @classmethod
    def from_string(cls, s):
        """
        Constructs a tree by parsing the given string. Used to process the output of collins parser,
        i.e a tree written in a single line
        """
        nodes = cls.__splitAndCheckTreeString(s)
        order_nb = 0 # numbering the leaves (i.e. word order position)
        # stack holds the ancestors of the given node
        stack = [Tree(nodes[1], None, [])]
        idx = 2  # position within the nodes list
        while len(stack) > 0:
            if idx >= len(nodes):
                raise NameError("ill-formed tree: didn't finish")
            n = nodes[idx]
            if n == '(':
                chlds = stack[-1].children  # children of the parent node
                if nodes[idx+2] == '(':
                    # internal node:
                    newN = Tree(nodes[idx+1], stack[-1], [])
                    chlds.append(newN)
                    stack.append(newN)
                    idx += 2
                else:
                    # leaf and its POS tag:
                    newN = Tree(nodes[idx+1], stack[-1], [Tree(nodes[idx+2])], order_nb=order_nb)
                    newN.children[0].parent = newN
                    chlds.append(newN)
                    idx += 4
                    order_nb += 1
            elif n == ')':
                if len(stack) == 1:
                    break
                else:
                    stack.pop()
                    idx += 1
            else:
                print(s)
                print([node.val for node in stack])
                print(n)
                raise NameError("ill-formed tree")

        return cls(stack[0].val, None, stack[0].children)

    @classmethod
    def __splitAndCheckTreeString(cls, s):
        """Used for parsing string. Splits the given string and checks whether
        it can represent a PEN-style trees"""
        s = s.replace('\n', ' ')
        s = s.replace('(', ' ( ')
        s = s.replace(')', ' ) ')
        nodes = s.split()
        if nodes[0] != '(':
            print(s)
            raise NameError("Tree does not start with '('")
        if nodes[-1] != ')':
            print(s)
            raise NameError("Tree does not end with ')'")

        # join tokens with spaces in them (e.g. CD 800 855 934)
        cleared_nodes = []
        stack = []
        for n in nodes:
            stack.append(n)
            if n in ('(', ')'):
                if len(stack) > 2:
                    stack = [stack[0], ' '.join(stack[1:-1]), stack[-1]]
                cleared_nodes.extend(stack)
                stack = []

        return cleared_nodes

    def __iter__(self):
        class TreeIterator():
            def __init__(self, iter_stack):
                self.iter_stack = iter_stack

            def next(self):
                if not self.iter_stack:
                    raise StopIteration
                node = self.iter_stack.pop()
                self.iter_stack.extend(reversed(node.children))
                return node

        return TreeIterator([self])

    def insert_lefmost_child(self, new_leaf):
        i = self.get_leftmost_child().order_nb
        for n in self.get_root():
            if n.is_leaf() and n.order_nb >= i:
                n.order_nb += 1


        # self.val = val  # type: str
        # self.parent = parent  # type: Tree
        # self.children = children  # type: List[Tree]
        # self.order_nb = order_nb  # type: int  # zero-based
        #