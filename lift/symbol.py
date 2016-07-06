class Symbol(object):
    required_fields = ()

    def __init__(self, position, **kwargs):
        self.position = position

        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v


class Argument(Symbol):
    required_fields = ('type', 'shape')

class Monad(Symbol):
    required_fields = ('rank', 'expr')

class Dyad(Symbol):
    required_fields = ('rank', 'expr')


class Literal(Symbol):
    required_fields = ('shape', 'body')

class ApplyMonad(Symbol):
    required_fields = ('shape', 'op', 'rank', 'y')

class ApplyDyad(Symbol):
    required_fields = ('shape', 'op', 'rank', 'x', 'y')

class Constant(Symbol):
    required_fields = ('shape', 'value')

class AccMonad(Symbol):
    required_fields = ('shape', 'acc', 'v', 'operand')

class AccDyad(Symbol):
    required_fields = ('shape', 'acc', 'v', 'operand')


class Table(object):

    def __init__(self, default):
        self.next_var_id = 0
        self.declarations = {}
        self.vars = {}
        self.symbols = default.copy()
        self.use_graphs = {}
        self.grads = {}

    def check_shape(self, name):
        v = self.symbols[name]
        assert self.declarations[name].shape == self.vars[v].shape, "shape of symbol '%s' does not match with declaration" % (name,)

    def add_declaration(self, name, value):
        assert name not in self.declarations, "'%s' already declared" % (name,)
        self.declarations[name] = value

        if name in self.symbols:
            self.check_shape(name)

    def add_var(self, value):
        name = "v"+str(self.next_var_id)
        self.next_var_id += 1
        self.vars[name] = value
        return name

    def itervars(self):
        for i in xrange(self.next_var_id):
            yield "v" + str(i)

    def add_symbol(self, name, value):
        assert name not in self.symbols, "symbol '%s' already defined" % (name,)
        self.symbols[name] = value

        if name in self.declarations:
            self.check_shape(name)

    def get_symbol(self, name):
        v = self.symbols.get(name, None)
        if v is None:
            d = self.declarations.get(name, None)
            assert d is not None, "'%s' is not defined" % (name,)
            assert d.type == "in", "'%s' is not input" % (name,)
            v = self.add_var(d)
            self.add_symbol(name, v)
        return v

    def get_use_graph(self, v):
        graph = self.use_graphs.get(v, None)
        if graph is None:
            assert self.vars[v].shape == ()
            graph = {}

            visited = set()
            queue = [v]

            while queue:
                u = queue.pop(0)
                if u in visited:
                    continue
                visited.add(u)
                var = self.vars[u]

                if isinstance(var, ApplyMonad):
                    queue.append(var.y)
                    graph[var.y] = graph.get(var.y,())+((u,'y'),)
                elif isinstance(var, ApplyDyad):
                    queue.append(var.x)
                    queue.append(var.y)
                    graph[var.x] = graph.get(var.x,())+((u,'x'),)
                    graph[var.y] = graph.get(var.y,())+((u,'y'),)
                elif isinstance(var, Literal):
                    pass
                elif isinstance(var, Argument):
                    pass
                else:
                    raise NotImplementedError

            self.use_graphs[v] = graph

        return graph


    def get_acc(self, acc, v, operand):
        val = self.vars[v]
        shape = self.vars[getattr(val, operand)].shape

        if isinstance(val, ApplyMonad):
            return self.add_var(
                AccMonad(
                    None,
                    shape = shape,
                    acc = acc,
                    v = v,
                    operand = operand
                )
            )
        elif isinstance(val, ApplyDyad):
            return self.add_var(
                AccDyad(
                    None,
                    shape = shape,
                    acc = acc,
                    v = v,
                    operand = operand
                )
            )
        else:
            raise NotImplementedError


    def get_grad(self, v, w):
        grad = self.grads.get((v,w), None)
        if grad is not None:
            return grad

        if v == w:
            return self.add_var(
                Constant(
                    None,
                    shape=(),
                    value=(1.0,)))

        graph = self.get_use_graph(v)
        assert w in graph

        u, operand = graph[w][0]
        result = self.get_acc(self.get_grad(v,u), u, operand)

        for u, operand in graph[w][1:]:
            result2 = self.get_acc(self.get_grad(v,u), u, operand)
            result = self.add_var(
                ApplyDyad(
                    None,
                    shape = self.vars[result].shape,
                    op = self.symbols['+'],
                    rank = (),
                    x = result,
                    y = result2))

        self.grads[(v,w)] = result
        return result
