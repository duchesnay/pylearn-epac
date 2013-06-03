"""
Base Workflow node plus keys manipulation utilities.

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""


import re
import sys
import numpy as np
from abc import abstractmethod
import ast
from epac.utils import _list_union_inter_diff, _list_indices
from epac.utils import _list_of_dicts_2_dict_of_lists
from epac.stores import StoreMem
from epac.results import ResultSet
from epac.configuration import conf, debug


## ================================= ##
## == Key manipulation utils      == ##
## ================================= ##

def key_push(key, basename):
    if key and basename:
        return key + conf.KEY_PATH_SEP + basename
    else:
        return key or basename

def key_pop(key, index=-1):
    """Split the key into head / tail.

    index: int
        index=-1 (default) tail = last item
        index=0  head = first item
    Example
    -------
    >>> key = 'CV/CV(nb=0)/SelectKBest/LDA'
    >>> print key_pop(key, index=-1)
    ('CV/CV(nb=0)/SelectKBest', 'LDA')
    >>> print key_pop(key, index=0)
    ('CV', 'CV(nb=0)/SelectKBest/LDA')
    """
    if index == 0:
        i = key.find(conf.KEY_PATH_SEP)
        head = key[:i]
        tail = key[(i + 1):]
    else:
        i = key.rfind(conf.KEY_PATH_SEP) + 1
        tail = key[i:]
        head = key[:(i - 1)]
    return head, tail


_match_args_re = re.compile(u'([^(]+)\(([^)]+)\)')


def key_split(key, eval_args=False):
    """ Split the key into signatures.

    Parameters
    ----------

    key: str

    eval_args: boolean
      If true "un-string" the parameters, and return (name, argument) tuples:
      [(item_name1, [[argname, value], ...]), ...]
    Examples
    --------
    >>> key_split(key='Methods/SelectKBest(k=1)/SVC(kernel=linear,C=1)')
    ['Methods', 'SelectKBest(k=1)', 'SVC(kernel=linear,C=1)']
    >>> key_split(key='Methods/SelectKBest(k=1)/SVC(kernel=linear,C=1)', eval_args=True)
    [('Methods',), ('SelectKBest', [['k', 1]]), ('SVC', [['kernel', 'linear'], ['C', 1]])]
    """
    signatures = [signature for signature in key.split(conf.KEY_PATH_SEP)]
    if eval_args:
        return [signature_eval(signature) for signature in signatures]
    else:
        return signatures


def signature_eval(signature):
    m = _match_args_re.findall(signature)
    if m:
        name = m[0][0]
        argstr = m[0][1]
        args = [argstr.split("=") for argstr in argstr.split(",")]
        for arg in args:
            try:
                arg[1] = ast.literal_eval(arg[1])
            except ValueError:
                pass
        return(name, args)
    else:
        return(signature, )

## ============================================== ##
## == down-stream data-flow manipulation utils == ##
## ============================================== ##

def xy_split(Xy):
    """Split Xy into two dictonaries. If input dictonnary whas not build
    with xy_merge(Xy1, Xy2) then return twice the input
    dictonnary.

    Parameters
    ----------
    Xy: dict

    Returns
    -------
    dict1, dict2 : splited dictionaries

    Example
    -------
    >>> xy_merged = xy_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    >>> print xy_merged
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    >>> print xy_split(xy_merged)
    ({'a': 1, 'b': 2}, {'a': 33, 'c': 55, 'b': 44})
    >>> print xy_split(dict(a=1, b=2))
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
    """
    keys1 = [key1 for key1 in Xy if (str(key1).find("__1__%") == 0)]
    keys2 = [key2 for key2 in Xy if (str(key2).find("__2__%") == 0)]
    if not keys1 and not keys2:
        return Xy, Xy
    if keys1 and keys2:
        Xy1 = {key1.replace("__1__%", ""):
                        Xy[key1] for key1 in keys1}
        Xy2 = {key2.replace("__2__%", ""):
                        Xy[key2] for key2 in keys2}
        return Xy1, Xy2
    raise KeyError("data-flow could not be splitted")


def xy_merge(Xy1, Xy2):
    """Merge two dict avoiding keys collision.

    Parameters
    ----------
    Xy1: dict
    Xy2: dict

    Returns
    -------
    dict : merged dictionary

    Example
    -------
    >>> xy_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    """
    Xy1 = {"__1__%" + str(k): Xy1[k] for k in Xy1}
    Xy2 = {"__2__%" + str(k): Xy2[k] for k in Xy2}
    Xy1.update(Xy2)
    return Xy1


## ======================================= ##
## == Workflow Node base abstract class == ##
## ======================================= ##

class BaseNode(object):
    """WorkFlow Node base abstract class"""

    def __init__(self):
        self.parent = None
        self.children = list()
        self.store = None
        # The Key is the concatenation of nodes signatures from root to
        # Leaf.
        # Arguments are used to avoid collisions between keys.
        # In downstream flow collisions should always be avoided, so if
        # several children have the same name, use argument to sign the node.
        # In upstream flow collisions lead to aggregation of children node
        # with the same signature.
        self.signature_args = None  # dict of args to build the node signature
        self.reducer = None

    def __repr__(self):
        return self.get_key()

    # --------------------- #
    # -- Tree operations -- #
    # --------------------- #

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_root(self):
        """Leaves iterator"""
        curr = self
        while True:
            if not curr.parent:
                return curr
            curr = curr.parent

    def walk_nodes(self):
        """Node iterator"""
        yield self
        if self.children:
            for child in self.children:
                for yielded in child.walk_nodes():
                    yield yielded

    def walk_leaves(self):
        """Leaves iterator"""
        if not self.children:
            yield self
        else:
            for child in self.children:
                for yielded in child.walk_leaves():
                    yield yielded

    def get_leftmost_leaf(self):
        """Return the left most leaf of a tree"""
        return self if not self.children else \
            self.children[0].get_leftmost_leaf()

    def get_rightmost_leaf(self):
        """Return the left most leaf of a tree"""
        return self if not self.children else \
            self.children[-1].get_rightmost_leaf()

    def get_node(self, key=None, regexp=None, stop_first_match=False):
        """Return a node given a key or a list of nodes given regexp

        Parameters
        ----------
        key: str

        regexp: str
            string with wild-cards: "*" to allow several matches.

        Example
        -------
        >>> from epac import CV, Grid, Pipe
        >>> from sklearn.svm import SVC
        >>> from sklearn.lda import LDA
        >>> from sklearn.feature_selection import SelectKBest
        >>> y = [1, 1, 2, 2]
        >>> wf = CV(Grid(*[Pipe(SelectKBest(k=k), SVC()) \
        ...     for k in [1, 5]]), n_folds=2, y=y)
        # List all leaves keys
        >>> for n in wf:
        ...     print n.get_key()
        ...
        CV/CV(nb=0)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=0)/Grid/SelectKBest(k=5)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=5)/SVC
        # Get a single node using excat key match
        >>> wf.get_node(key="CV/CV(nb=0)/Grid/SelectKBest(k=1)").get_key()
        'CV/CV(nb=0)/Grid/SelectKBest(k=1)'
        # Get several nodes using wild cards
        >>> for n in wf.get_node(regexp="CV/*"):
        ...         print n.get_key()
        ...
        CV/CV(nb=0)
        CV/CV(nb=1)
        >>> for n in wf.get_node(regexp="*CV/CV(*)/*/*/SVC"):
        ...     print n.get_key()
        ...
        CV/CV(nb=0)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=0)/Grid/SelectKBest(k=5)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=5)/SVC
        """
        if key:
            if key == self.get_key():
                return self
            for child in self.children:
                if key.find(child.get_key()) == 0:
                    return child.get_node(key)
        elif regexp:
            if isinstance(regexp, str):
                regexp = re.compile(regexp.replace("*", ".*").\
                             replace("(", "\(").replace(")", "\)"))
            if regexp.match(self.get_key()):
                return [self]
            nodes = list()
            for child in self.children:
                node = child.get_node(key=None, regexp=regexp,
                                      stop_first_match=stop_first_match)
                if node and stop_first_match:
                    return node[0] if isinstance(node, list) else node
                nodes += node
            return nodes
        else:
            raise ValueError("Provide at least a key for exact match"
            "or a regexp for wild card matches")

#    def get_path_from_root(self):
#        if self.parent is None:
#            return [self]
#        return self.parent.get_path_from_root() + [self]

    def get_path_from_root(self):
        """Get path iterator from root.

        See also
        --------
        get_path_from_node(node)
        """
        return self.get_path_from_node(node=self.get_root())

    def get_path_from_node(self, node):
        """
        Get path iterator from node to self.

        Example
        -------
        >>> from epac import Perms, CV, Pipe, Methods
        >>> from sklearn.lda import LDA
        >>> from sklearn.svm import SVC
        >>> from sklearn.feature_selection import SelectKBest
        >>> root = Perms(CV(Pipe(SelectKBest(k=2), Methods(LDA(), SVC()))))
        >>> leaf = root.get_leftmost_leaf()
        >>> print leaf
        Perms/Perm(nb=0)/CV/CV(nb=0)/SelectKBest/Methods/LDA
        >>> node = root.children[2].children[0]
        >>> print node
        Perms/Perm(nb=2)/CV
        >>> print [n.get_signature() for n in leaf.get_path_from_node(node=node)]
        ['CV', 'CV(nb=0)', 'SelectKBest', 'Methods', 'LDA']
        >>> print [n.get_signature() for n in leaf.get_path_from_root()]
        ['Perms', 'Perm(nb=2)', 'CV', 'CV(nb=0)', 'SelectKBest', 'Methods', 'LDA']
        """
        key = self.get_key()
        parent_key = node.get_key()
        path_key = key.replace(parent_key, "").lstrip(conf.KEY_PATH_SEP)
        key_parts = key_split(path_key)
        #idx = len(key_parts) - 1
        curr = self
        # Check if node can be found in parents
        while curr and curr.get_key() != parent_key:
            curr = curr.parent
        if not curr or curr is not node:
            raise ValueError('Parent node could not be found in tree')
        # Go down from node to self
        yield curr
        while key_parts:
            signature = key_parts.pop(0)
            for child in curr.children:
                if child.get_signature() == signature:
                    break
            curr = child
            yield curr

#    def get_path_from_node(self, node):
#        if self is node:
#            return [self]
#        return self.parent.get_path_from_node(node) + [self]

    # --------------------- #
    # -- Key             -- #
    # --------------------- #

    def get_key(self, nb=1):
        """Return primary or intermediate key.

        Primary key is a unique identifier of a node in the tree, it is
        used to store leaf outputs at the end of the downstream flow.
        Intermediate key identify upstream results. Collisions between
        intermediate trig an aggregation of all results having the same key.

        Argument
        --------
        nb: int
            1 (default) return the primary key (ds data-flow).
            2 return the intermediate key (us data-flow).
        """
        if not self.parent:
            return self.get_signature(nb=nb)
            #return key_push(self.store, self.get_signature(nb=nb))
        elif not hasattr(self.parent, "get_key"):
            return key_push(str(self.parent), self.get_signature(nb=nb))
        else:
            return key_push(self.parent.get_key(nb=nb),
                            self.get_signature(nb=nb))

    def get_signature(self, nb=1):
        """The signature of the current Node, used to build the key.

        By default primary and intermediate signatures are identical.
        Aggregation behavior of results during up-stream can be controled here:
        Redefine this method to create key collision and aggregation when nb=2.
        """
        if not self.signature_args:
            return self.__class__.__name__
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.__class__.__name__ + args_str

    def get_signature_args(self):
        return self.signature_args

    @abstractmethod
    def get_parameters(self):
        """Return the state of the object"""

    def get_store(self):
        """Return the first store found on the path to tree root. If no store
        has been defined create one on the tree root and return it."""
        curr = self
        while True:
            if curr.store:
                return curr.store
            if not curr.parent:
                curr.store = StoreMem()
                return curr.store
            curr = curr.parent

    def stats(self, group_by="key", sort_by="count"):
        """Statistics on the workflow
            Parameters
            ----------
            group_by: str
                Group by "key" or class name: "class" (default "key")
            sort_by: str
                Sort by "count" or "size" (default "count")
        """
        stat_dict = dict()
        for n in self.walk_nodes():
            if group_by == "class":
                name = n.__class__.__name__
            else:
                name = n.get_signature()
            if not (name in stat_dict):
                stat_dict[name] = [0, 0]
            stat_dict[name][0] += 1
            stat_dict[name][1] += sys.getsizeof(n)
        stat = [(name, stat_dict[name][0], stat_dict[name][1]) for name in
                stat_dict]
        idx = 2 if sort_by == "size" else 1
        order = np.argsort([s[idx] for s in stat])[::-1]
        return [stat[i] for i in order]

    # ------------------------------------------ #
    # -- Top-down data-flow operations        -- #
    # ------------------------------------------ #

    def top_down(self, func_name, recursion=True, **Xy):
        """Top-down data processing method

            This method does nothing more that recursively call
            parent/children func_name. Most of time, it should be re-defined.

            Parameters
            ----------
            func_name: str
                the name of the function to be called
            recursion: boolean
                if True recursively call parent/children func_name. If the
                current node is the root of the tree call the children.
                This way the whole tree is executed.
                If it is a leaf, then recursively call the parent before
                being executed. This a pipeline made of the path from the
                leaf to the root is executed.
            **Xy: dict
                the keyword dictionnary of data-flow

            Return
            ------
            A dictionnary of processed data
        """
        if conf.TRACE_TOPDOWN:
            print self.get_key(), func_name
        if debug.DEBUG:
            debug.current = self
            debug.Xy = Xy
        func = getattr(self, func_name)
        Xy = func(recursion=False, **Xy)
        #Xy = self.transform(**Xy)
        if recursion and self.children:
            # Call children func_name down to leaves
            ret = [child.top_down(func_name=func_name, recursion=recursion,
                            **Xy) for child in self.get_children_top_down()]
            Xy = ret[0] if len(ret) == 1 else ret
        return Xy

    def fit(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        return Xy

    def transform(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        return Xy

    def predict(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **Xy)
        return Xy

    def fit_predict(self, recursion=True, **Xy):
        if recursion:  # fit_predict was called in a top-down recursive context
            return self.top_down(func_name="fit_predict", recursion=recursion,
                                 **Xy)
        Xy_train, Xy_test = xy_split(Xy)
        Xy_train = self.fit(recursion=False, **Xy_train)
        Xy_test = self.predict(recursion=False, **Xy_test)
        if self.children:
            return xy_merge(Xy_train, Xy_test)
        else:
            return Xy_test

    def get_children_top_down(self):
        """Return children during the top-down exection."""
        return self.children

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #

    def bottum_up2(self, store_results=True):
        # Terminaison (leaf) node return results
        if not self.get_children_bottum_up():
            return self.load_state(name="results")
        # 1) Build sub-aggregates over children
        children_results = [child.bottum_up(store_results=False) for
            child in self.get_children_bottum_up()]
        results = ResultSet(children_results)
        for result in results:
            result["key"] = key_push(self.get_signature(), result["key"])
        #
        if debug.DEBUG:
            debug.current = self
        if len(children_results) == 1:
            if store_results:
                self.save_state(state=children_results[0], name="results")
            return children_results[0]
        # 2) Test collision between intermediary keys
        keys_all = list()
        keys_set = set()
        for r in children_results:
            keys_all += r.keys()
            keys_set.update(r.keys())
        # 3) No collision: merge results in a lager dict an return
        if len(keys_set) == len(keys_all):
            merge = ResultSet()
            [merge.update(item) for item in children_results]
            if store_results:
                self.save_state(state=merge, name="results")
            return merge
        # 4) Collision occurs
        # Aggregate (stack) all children results with identical
        # intermediary key, by stacking them according to
        # argumnents. Ex.: stack for a CV stack folds, for a Grid
        children_args = [child.get_signature_args() for child in self.children]
        _, arg_names, diff_arg_names = _list_union_inter_diff(*[d.keys()
                                                for d in children_args])
        if diff_arg_names:
            raise ValueError("Children have different arguments name")
        sub_arg_names, sub_arg_values, results =\
            self._stack_results_over_argvalues(arg_names, children_results,
                                          children_args)
        # Reduce results if there is a reducer
        if self.reducer:
            results = {key2: self.reducer.reduce(result=results[key2]) for
                key2 in results}
        if store_results:
            self.save_state(state=results, name="results")
        return results

    def _stack_results_over_argvalues(self, arg_names, children_results,
                                      children_args):
        if not arg_names:
            # Should correspond to a single results
            if len(children_results) != 1:
                raise ValueError("Many results were only one expected")
            return [], [], children_results[0]
        arg_name = arg_names[0]
        stack_arg = list()
        children_arg = [child_arg[arg_name] for child_arg in children_args]
        arg_values = list(set(sorted(children_arg)))
        for val in arg_values:
            children_select = _list_indices(children_arg, val)
            children_results_select = [children_results[i] for i
                                            in children_select]
            children_args_select = [children_args[i] for i in children_select]
            sub_arg_names, sub_arg_values, sub_stacked = \
                self._stack_results_over_argvalues(
                              arg_names=arg_names[1:],
                              children_results=children_results_select,
                              children_args=children_args_select)
            stack_arg.append(sub_stacked)
        stacked = _list_of_dicts_2_dict_of_lists(stack_arg,
                                                 axis_name=arg_name,
                                                 axis_values=arg_values)
        return arg_names, arg_values, stacked

    #reduce = bottum_up

    def get_children_bottum_up(self):
        """Return children during the bottum-up execution."""
        return self.children

    # -------------------------------- #
    # -- I/O persistance operations -- #
    # -------------------------------- #

    def save_state(self, state, name="default"):
        store = self.get_store()
        store.save(key_push(self.get_key(), name), state)

    def load_state(self, name="default"):
        return self.get_store().load(key_push(self.get_key(), name))

    def save_tree(self, store):
        """I/O (persistance) operation: save the whole tree: ie.: execution
        tree + stores.

        Parameters
        ----------
        store: Store

        See Also
        --------
        Store.load()
        """
        # Save execution tree without the stores
        stores = dict()
        for node in self._walk_true_nodes():
            if node.store:
                stores[node.get_key()] = node.store
                node.store = None
        store.save(key=conf.STORE_EXECUTION_TREE_PREFIX,
                   obj=self, protocol="bin")
        for key1 in stores:
            node = self.get_node(key1)
            node.store = stores[key1]
        # Save the stores
        for key1 in stores:
            print key1, stores[key1]
            store.save(key=key_push(key1, conf.STORE_STORE_PREFIX),
                       obj=stores[key1], protocol="bin")

    def save_node(self, store):
        """I/O (persistance) operation: save single node states ie.: store"""
        store.save(key=key_push(self.get_key(), conf.STORE_STORE_PREFIX),
                       obj=self.store, protocol="bin")

    def _walk_true_nodes(self):
        yield self
        if self.children:
            if isinstance(self.children, list):
                children = self.children
            else:
                children = [self.children[0]]
            for child in children:
                for yielded in child.walk_nodes():
                    yield yielded
