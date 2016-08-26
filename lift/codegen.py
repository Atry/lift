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
    'div': '/',
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


def build_ast(stmts, node):
    if node.get_type() == isl.ast_node_type.block:
        return [build_ast(stmts, c) for c in node_list(node.block_get_children())]
    elif node.get_type() == isl.ast_node_type.for_:
        return ("for",
            convert_expr(node.for_get_iterator()),
            convert_expr(node.for_get_init()),
            convert_expr(node.for_get_inc()),
            convert_expr(node.for_get_cond()),
            build_ast(stmts, node.for_get_body()))
    elif node.get_type() == isl.ast_node_type.if_:
        if node.if_has_else():
            return (
                "ifelse",
                convert_expr(node.if_get_cond()),
                build_ast(stmts, node.if_get_then()),
                build_ast(stmts, node.if_get_else()))
        else:
            return (
                "if",
                convert_expr(node.if_get_cond()),
                build_ast(stmts, node.if_get_then()))
    elif node.get_type() == isl.ast_node_type.user:
        name = node.user_get_expr().get_id().get_name()
        return stmts[name]
    else:
        raise NotImplementedError


def get_stmt(ctx, name):
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
    return stmt


def transform_var(bmap, iterator_map, build):
    pma = isl.PwMultiAff.from_map(isl.Map.from_union_map(iterator_map.apply_range(bmap)))
    expr = build.access_from_pw_multi_aff(pma)
    n = expr.get_op_n_arg()
    return ('element', expr.get_op_arg(0).get_id().get_name() , tuple(convert_expr(expr.get_op_arg(i)) for i in xrange(1,n)))


def transform_expr(expr, iterator_map, build):
    if expr[0] == 'var':
        return transform_var(expr[1], iterator_map, build)
    elif expr[0] == 'const':
        return expr
    elif expr[0] == 'call':
        return ('call', expr[1], tuple(transform_expr(e, iterator_map, build) for e in expr[2]))
    else:
        raise NotImplementedError


def transform_stmt(stmt, iterator_map, build):
    name = stmt[0].get_tuple_name(isl.dim_type.in_)

    return ("assign", transform_var(stmt[0], iterator_map, build),
            transform_expr(stmt[1], iterator_map, build), name)


def codegen(ctx, constraints):
    schedule = constraints.compute_schedule()
    stmts = {}

    def at_each_domain(node, build):
        if node.get_type() != isl.ast_node_type.user:
            return node

        expr = node.user_get_expr()
        assert expr.get_type() == isl.ast_expr_type.op
        assert expr.get_op_type() == isl.ast_op_type.call
        expr = expr.get_op_arg(0)
        assert expr.get_type() == isl.ast_expr_type.id

        name = expr.get_id().get_name()
        stmt = get_stmt(ctx, name)

        iterator_map = build.get_schedule().reverse()

        name = "S"+str(len(stmts))
        stmts[name] = transform_stmt(stmt, iterator_map, build)

        return isl.AstNode.alloc_user(isl.AstExpr.from_id(isl.Id.alloc(schedule.get_ctx(), name, None)))

    build = isl.AstBuild.alloc(ctx.isl_context)
    build, _ = build.set_at_each_domain(at_each_domain)
    node = build.node_from_schedule(schedule)

    return build_ast(stmts, node)


def get_schedule_constraints(ctx):
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

    proximity = validity.union(update_map.apply_range(update_map.reverse()))

    constraints = (
        isl.ScheduleConstraints.on_domain(domain)
        .set_validity(validity)
        .set_proximity(proximity)
    )

    return constraints
