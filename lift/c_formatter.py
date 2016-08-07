def get_shapes(table, ctx):
    vars = (
        ctx.input_arrays.keys()
        + ctx.output_arrays.keys()
        + ctx.intermediate_arrays.keys()
        + ctx.const_arrays.keys())

    return {v:table.vars[v].shape for v in vars}


def format_arguments(table, ctx):
    vars = (
        ctx.input_arrays.keys()
        + ctx.output_arrays.keys())

    for v in vars:
        yield "    float %s%s"%(v, "".join("[%d]"%(s,) for s in table.vars[v].shape[::-1]))

def format_scalar_consts(table, ctx):
    for k,v in ctx.const_values.items():
        if len(table.vars[k].shape) != 0:
            continue

        yield "  float %s=%s;\n" % (k, v[0])


def format_scalar_vars(table, ctx):
    for k in ctx.intermediate_arrays:
        if len(table.vars[k].shape) != 0:
            continue

        yield "  float %s;\n" % (k,)


def format_consts(table, ctx):
    for k,v in ctx.const_values.items():
        shape = table.vars[k].shape[::-1]
        if len(shape) == 0:
            continue

        yield "  float %s%s={%s};\n" % (
            k,
            "".join("[%d]"%(s,) for s in shape),
            ",".join("%f"%(x,) for x in v))


def format_vars(table,ctx):
    for k in ctx.intermediate_arrays:
        shape = table.vars[k].shape[::-1]
        if len(shape) == 0:
            continue

        yield "  float %s%s;\n"%(k,"".join("[%d]"%(s,) for s in shape))


C_TEMPLATE = """#include <math.h>
#define inf (1.0/0.0)
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

void
{name}(
{arguments}){{
{scalar_consts}{scalar_vars}{consts}{vars}
#pragma scop
{body}
#pragma endscop
}}"""

def format_c(name, table, ctx, ast):

    return C_TEMPLATE.format(
        name = name,
        arguments = ",\n".join(format_arguments(table, ctx)),
        scalar_consts = "".join(format_scalar_consts(table,ctx)),
        scalar_vars = "".join(format_scalar_vars(table,ctx)),
        consts = "".join(format_consts(table,ctx)),
        vars = "".join(format_vars(table,ctx)),
        body = format_ast(table, ast)
    )


INT_FUN_TYPE = {
    '<': 'infix',
    '>': 'infix',
    '<=': 'infix',
    '>=': 'infix',
    '==': 'infix',
    '!=': 'infix',
    '&&': 'infix',
    '||': 'infix',
    '+': 'infix',
    '-': 'infix',
    '*': 'fun2',
    '/': 'infix',
    '_': "prefix",
    '%': 'infix',
    'max': 'fun2',
    'min': 'fun2',
}

INT_FUN_FMT = {
    '_': '-',
}

def format_int_call(fun, args):
    t = INT_FUN_TYPE[fun]
    if t == 'infix':
        return "({} {} {})".format(format_int_expr(args[0]), INT_FUN_FMT.get(fun,fun), format_int_expr(args[1]))
    elif t == 'prefix':
        return "({} {})".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun1':
        return "({}({}))".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]))
    elif t == 'fun2':
        return "({}({},{}))".format(INT_FUN_FMT.get(fun,fun), format_int_expr(args[0]), format_int_expr(args[1]))

    raise NotImplementedError


def format_int_expr(ast):
    if ast[0] == 'var':
        return '{}'.format(ast[1])
    elif ast[0] == 'int':
        return "%d"%(ast[1],)
    elif ast[0] == 'call':
        return format_int_call(ast[1], ast[2])

    raise NotImplementedError


def format_element(v, indices):
    return "%s%s"%(v, "".join("[%s]"%(format_int_expr(i),) for i in indices))


FUN_TYPE = {
    '+': 'infix',
    '-': 'infix',
    '*': 'infix',
    '/': 'infix',
    '_': 'prefix',
    '**': 'fun2',
    'exp': 'fun1',
    'log': 'fun1',
    '==': 'infix',
    'max': 'fun2',
    'min': 'fun2',
}

FUN_FMT = {
    '_': '-',
    '**': 'powf',
    'exp': 'expf',
    'log': 'logf',
    'max': 'fmax',
    'min': 'fmin',
}

def format_call(table, fun, args):
    t = FUN_TYPE[fun]
    if t == 'infix':
        return "({} {} {})".format(format_ast(table, args[0]), FUN_FMT.get(fun,fun), format_ast(table, args[1]))
    elif t == 'prefix':
        return "({} {})".format(FUN_FMT.get(fun,fun), format_ast(table, args[0]))
    elif t == 'fun1':
        return "({}({}))".format(FUN_FMT.get(fun, fun), format_ast(table, args[0]))
    elif t == 'fun2':
        return "({}({},{}))".format(FUN_FMT.get(fun, fun), format_ast(table, args[0]), format_ast(table, args[1]))

    raise NotImplementedError


def format_ast(table, ast):
    if isinstance(ast, list):
        return "".join(format_ast(table, node) for node in ast)
    if ast[0] == 'for':
        return "for(int {v} = {init}; {cond}; {v} = {v} + {inc}){{\n{body}}}\n".format(
            v = format_int_expr(ast[1]),
            init = format_int_expr(ast[2]),
            inc = format_int_expr(ast[3]),
            cond = format_int_expr(ast[4]),
            body = format_ast(table, ast[5]))
    elif ast[0] == 'if':
        return "if({cond}){{\n{then}}}\n".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(table, ast[2]))
    elif ast[0] == 'ifelse':
        return "if({cond}){{\n{then}}}\nelse{{\n{else_}}}".format(
            cond = format_int_expr(ast[1]),
            then = format_ast(table, ast[2]),
            else_ = format_ast(table, ast[3]))
    elif ast[0] == 'assign':
        assert ast[1][0] == 'element'
        shape = table.vars[ast[1][1]].shape
        if len(shape) == 0:
            assert len(ast[1][2]) == 0
            return "{} = {};\n".format(
                ast[1][1],
                format_ast(table, ast[2]))
        return "{} = {};\n".format(
            format_element(ast[1][1], ast[1][2]),
            format_ast(table, ast[2]))
    elif ast[0] == 'call':
        return format_call(table, ast[1], ast[2])
    elif ast[0] == 'element':
        shape = table.vars[ast[1]].shape
        if len(shape) == 0:
            assert len(ast[2]) == 0
            return ast[1]
        else:
            return "({})".format(format_element(ast[1], ast[2]))
    elif ast[0] == 'const':
        return "%f"%(ast[1],)
    elif ast[0] == 'var':
        return ast[1]
    else:
        raise NotImplementedError
