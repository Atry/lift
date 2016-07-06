import islpy as isl

def node_list(l):
    r = []
    l.foreach(lambda e: r.append(e))
    return r


OP = {
    'le': '<=',
    'ge': '>=',
    'lt': '<',
    'gt': '>',
    'max': 'max',
    'min': 'min',
    'sub': '-',
    'minus': '_',
    'add': '+',
    'eq': '==',
    'and_': '&&',
    'pdiv_r': '%',
    'zdiv_r': '%',
}

def convert_expr(expr):
    if expr.get_type() == isl.ast_expr_type.id:
        return ('var', expr.get_id().get_name())
    elif expr.get_type() == isl.ast_expr_type.int:
        return ('int', expr.get_val().to_python())
    elif expr.get_type() == isl.ast_expr_type.op:
        return ('call',
                OP[isl.ast_op_type.find_value(expr.get_op_type())],
                tuple(convert_expr(expr.get_op_arg(i)) for i in xrange(expr.get_op_n_arg())))
    else:
        raise NotImplementedError


def to_basic_maps(map):
    maps = []
    map.foreach_basic_map(lambda x: maps.append(x))
    return maps


def transform_var(bmap, args):
    bmap, = to_basic_maps(bmap.affine_hull())
    n_out = bmap.dim(isl.dim_type.out)
    n_in_ = bmap.dim(isl.dim_type.in_)

    out_args = [None for i in xrange(n_out)]

    for constraint in bmap.get_constraints():
        if not constraint.is_equality():
            continue
        if not constraint.involves_dims(isl.dim_type.out, 0, n_out):
            continue

        index = ('int', -constraint.get_constant_val().to_python())

        for i, v in enumerate(constraint.get_coefficient_val(isl.dim_type.in_, i) for i in xrange(n_in_)):
            if v.is_zero():
                continue

            index1 = args[i]

            if not v.is_negone():
                index1 = ('call', '*', (('int', -v.to_python()), index1))

            if index == ('int', 0):
                index = index1
            elif v.is_one():
                index = ('call', '-', (index, index1[2][1]))
            else:
                index = ('call', '+', (index, index1))

        (out_index, out_coeff), = (
            (i, v.to_python())
            for i, v in enumerate(constraint.get_coefficient_val(isl.dim_type.out, i) for i in xrange(n_out))
            if not v.is_zero())

        if out_coeff != 1:
            index = ('call', '/', (index, ('int', out_coeff)))

        out_args[out_index] = index

    assert all(a is not None for a in out_args)
    return ('element', bmap.get_tuple_name(isl.dim_type.out), tuple(out_args))


def transform_expr(expr, args):
    if expr[0] == 'var':
        return transform_var(expr[1], args)
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return ('call', expr[1], tuple(transform_expr(e, args) for e in expr[2]))
    else:
        raise NotImplementedError


def convert_stmt(ctx, expr):
    assert isl.ast_op_type.find_value(expr.get_op_type()) == 'call'
    name = expr.get_op_arg(0).get_id().get_name()
    args = tuple(convert_expr(expr.get_op_arg(i)) for i in xrange(1, expr.get_op_n_arg()))

    if name in ctx.def_stmts:
        stmt = ctx.def_stmts[name]
    elif name in ctx.init_stmts:
        stmt = ctx.init_stmts[name]
    elif name in ctx.update_stmts:
        stmt = ctx.update_stmts[name]
    elif name in ctx.fini_stmts:
        stmt = ctx.fini_stmts[name]
    else:
        raise NotImplementedError

    return ("assign", transform_var(stmt[0], args),
            transform_expr(stmt[1], args))


def build_ast(ctx, node):
    if node.get_type() == isl.ast_node_type.block:
        return [build_ast(ctx, c) for c in node_list(node.block_get_children())]
    elif node.get_type() == isl.ast_node_type.for_:
        return ("for",
            convert_expr(node.for_get_iterator()),
            convert_expr(node.for_get_init()),
            convert_expr(node.for_get_inc()),
            convert_expr(node.for_get_cond()),
            build_ast(ctx, node.for_get_body()))
    elif node.get_type() == isl.ast_node_type.if_:
        if node.if_has_else():
            return (
                "ifelse",
                convert_expr(node.if_get_cond()),
                build_ast(ctx, node.if_get_then()),
                build_ast(ctx, node.if_get_else()))
        else:
            return (
                "if",
                convert_expr(node.if_get_cond()),
                build_ast(ctx, node.if_get_then()))
    elif node.get_type() == isl.ast_node_type.user:
        return convert_stmt(ctx, node.user_get_expr())
    else:
        raise NotImplementedError


def get_schedule_map(ctx):
    def_map = ctx.def_stmts.get_assign_map().union(ctx.fini_stmts.get_assign_map())
    init_map = ctx.init_stmts.get_assign_map()
    update_map = ctx.update_stmts.get_assign_map()

    use_map = ctx.get_use_map()

    domain = (
        def_map.domain()
        .union(init_map.domain())
        .union(update_map.domain()))
    validity = (
        def_map.apply_range(use_map.reverse())
        .union(init_map.apply_range(update_map.reverse()))
        .union(update_map.apply_range(def_map.reverse())))

    constraints = (
        isl.ScheduleConstraints.on_domain(domain)
        .set_validity(validity)
        .set_coincidence(validity)
        .set_proximity(validity)
    )
    schedule = constraints.compute_schedule()
    return schedule.get_map().intersect_domain(domain)


def codegen(ctx):
    schedule_map = get_schedule_map(ctx)
    node = isl.AstBuild.alloc(ctx.isl_context).ast_from_schedule(schedule_map)
    return build_ast(ctx, node)
