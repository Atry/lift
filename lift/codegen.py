import re
import islpy as isl
from .utils import Counter
from .isl_utils import to_sets, to_list, list_to_multival


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
    'mul': '*',
    'div': '/',
    'pdiv_q': '/',
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
        return [build_ast(stmts, c) for c in to_list(node.block_get_children())]
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
        match = re.match(r'kernel\[(\d+)\]', name)
        if match is None:
            return stmts[name]

        kernel_id = int(match.group(1))
        return ("kernel", kernel_id)
    elif node.get_type() == isl.ast_node_type.mark:
        return (
            "mark",
            node.mark_get_id().get_name(),
            build_ast(stmts, node.mark_get_node())
        )
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
            transform_expr(stmt[1], iterator_map, build))


class ArrayInfo(object):

    def __init__(self, has_cpu_buffer, gpu_up_to_date):
        self.has_cpu_buffer = has_cpu_buffer
        self.has_gpu_buffer = False
        self.cpu_up_to_date = True
        self.gpu_up_to_date = gpu_up_to_date
        self.last_gpu_write = None
        self.last_gpu_access = None

class ArrayState(object):

    def __init__(self, ctx, assign_map, use_map, arrays, domains, kernels):
        self.ctx = ctx
        self.assign_map = assign_map
        self.use_map = use_map
        self.arrays = arrays
        self.domains = domains
        self.kernels = kernels
        self.last_kernel = None
        self.acc = isl.UnionSet("{ }", ctx.isl_context)

    def on_domain(self, name):
        self.acc = self.acc.union(domains[name])

    def cpu_access(self, v, write=False):
        array = self.arrays[v]
        array.has_cpu_buffer = True

        if not array.cpu_up_to_date:
            self.kernels[array.last_gpu_write].copy_outs.add(v)
            array.cpu_up_to_date = True

        if write:
            array.gpu_up_to_date = False

    def gpu_acccess(self, v, kernel_id, write=False):
        kernel = self.kernels[kernel_id]
        array = self.arrays[v]

        if not array.has_gpu_buffer:
            kernel.creates.add(v)
            array.has_gpu_buffer = True

        if not array.gpu_up_to_date:
            kernel.copy_ins.add(v)
            array.gpu_up_to_date = True

        array.last_gpu_access = kernel_id

        if write:
            array.last_gpu_write = kernel_id
            array.cpu_up_to_date = False


    def on_kernel(self, kernel_id):
        last_kernel = self.last_kernel
        self.last_kernel = kernel_id

        if not self.acc.is_empty():
            if last_kernel is not None:
                self.kernels[last_kernel].finish = True

            domain = self.acc
            self.acc = isl.UnionSet("{ }", self.ctx.isl_context)

            writes = assign_map.intersect_domain(domain).range()
            writes = list({ s.get_tuple_name() for s in to_sets(writes) })
            reads = use_map.intersect_domain(domain).range()
            reads = list({ s.get_tuple_name() for s in to_sets(reads) })

            for v in reads:
                self.cpu_access(v, write=False)

            for v in writes:
                self.cpu_access(v, write=True)

        kernel = self.kernels[kernel_id]

        for v in kernel.reads:
            self.gpu_acccess(v, kernel_id, write=False)

        for v in kernel.writes:
            self.gpu_acccess(v, kernel_id, write=True)


def accumulate_domain(state, node):
    if node.get_type() == isl.ast_node_type.block:
        for c in to_list(node.block_get_children()):
            accumulate_domain(state, c)
    elif node.get_type() == isl.ast_node_type.for_:
        accumulate_domain(state, node.for_get_body())
    elif node.get_type() == isl.ast_node_type.if_:
        accumulate_domain(state, node.if_get_then())
        if node.if_has_else():
            accumulate_domain(state, node.if_get_else())
    elif node.get_type() == isl.ast_node_type.user:
        name = node.user_get_expr().get_id().get_name()
        match = re.match(r'kernel\[(\d+)\]', name)
        assert match is None

        state.on_domain(name)
    elif node.get_type() == isl.ast_node_type.mark:
        accumulate_domain(state, node.mark_get_node())
    else:
        raise NotImplementedError


def collect_array_info(state, node):
    if node.get_type() == isl.ast_node_type.block:
        for c in to_list(node.block_get_children()):
            collect_array_info(state, c)
    elif node.get_type() == isl.ast_node_type.for_:
        accumulate_domain(state, node.for_get_body())
    elif node.get_type() == isl.ast_node_type.if_:
        accumulate_domain(state, node.if_get_then())
        if node.if_has_else():
            accumulate_domain(state, node.if_get_else())
    elif node.get_type() == isl.ast_node_type.user:
        name = node.user_get_expr().get_id().get_name()
        match = re.match(r'kernel\[(\d+)\]', name)
        if match is None:
            accumulate_domain(state, node)
        else:
            kernel_id = int(match.group(1))
            state.on_kernel(kernel_id)
    elif node.get_type() == isl.ast_node_type.mark:
        collect_array_info(state, node.mark_get_node())
    else:
        raise NotImplementedError



def codegen(ctx, schedule, kernels=None):
    names = Counter("E%d")
    stmts = {}
    domains = {}

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

        schedule = build.get_schedule()
        iterator_map = schedule.reverse()

        name = names.next()
        stmts[name] = transform_stmt(stmt, iterator_map, build)
        domains[name] = schedule.domain()

        return isl.AstNode.alloc_user(isl.AstExpr.from_id(isl.Id.alloc(ctx.isl_context, name, None)))

    def after_mark(node, build):
        id = node.mark_get_id()
        match = re.match(r'kernel\[(\d+)\]', id.get_name())
        if match is None:
            return node

        kernel_id = int(match.group(1))
        kernels[kernel_id].ast = node.mark_get_node()
        return isl.AstNode.alloc_user(isl.AstExpr.from_id(id))

    build = isl.AstBuild.alloc(ctx.isl_context)
    # must hold handle here
    build, _h1 = build.set_after_each_mark(after_mark)
    build, _h2 = build.set_at_each_domain(at_each_domain)
    node = build.node_from_schedule(schedule)

    if kernels is not None:
        assign_map = (
            ctx.init_stmts.get_assign_map()
            .union(ctx.fini_stmts.get_assign_map())
            .union(ctx.update_stmts.get_assign_map())
            .union(ctx.def_stmts.get_assign_map())
        )

        use_map = ctx.get_use_map()

        for kernel in kernels.itervalues():
            kernel.ast = build_ast(stmts, kernel.ast)
            writes = assign_map.intersect_domain(kernel.domain).range()
            reads = use_map.intersect_domain(kernel.domain).range()
            accesses = writes.union(reads)
            kernel.reads = list({ s.get_tuple_name() for s in to_sets(reads) })
            kernel.writes = list({ s.get_tuple_name() for s in to_sets(writes) })
            kernel.arrays = list({ s.get_tuple_name() for s in to_sets(accesses) })
            kernel.finish = False


            # create buffer
            kernel.creates = set()
            # copy to gpu
            kernel.copy_ins = set()
            # copy from gpu
            kernel.copy_outs = set()
            # release buffer
            kernel.releases = set()


        # XXX: we assume no conditional kernel here
        arrays = {}

        assert not ctx.const_arrays

        for v in ctx.input_arrays:
            arrays[v] = ArrayInfo(has_cpu_buffer=True, gpu_up_to_date=False)

        for v in ctx.output_arrays:
            arrays[v] = ArrayInfo(has_cpu_buffer=True, gpu_up_to_date=True)

        for v in ctx.intermediate_arrays:
            arrays[v] = ArrayInfo(has_cpu_buffer=False, gpu_up_to_date=True)

        state = ArrayState(ctx, assign_map, use_map, arrays, domains, kernels)

        collect_array_info(state, node)

        if state.last_kernel is not None:
            kernels[state.last_kernel].finish = True

        for v in ctx.output_arrays:
            state.cpu_access(v, write=False)

        for v, a in arrays.iteritems():
            if not a.has_gpu_buffer:
                continue
            kernels[a.last_gpu_access].releases.add(v)

        kernels["arrays"] = [v for v in ctx.intermediate_arrays if arrays[v].has_cpu_buffer]

    ast = build_ast(stmts, node)
    return ast


def get_schedule_constraints(ctx):
    def_map = ctx.def_stmts.get_assign_map().union(ctx.fini_stmts.get_assign_map())

    init_map = ctx.init_stmts.get_assign_map()
    update_map = ctx.update_stmts.get_assign_map()

    use_map = ctx.get_use_map()

    domain = (
        def_map.domain()
        .union(init_map.domain())
        .union(update_map.domain())
    )

    validity = (
        def_map.apply_range(use_map.reverse())
        .union(init_map.apply_range(update_map.reverse()))
        .union(update_map.apply_range(def_map.reverse()))
    )

    coincidence = validity.union(update_map.apply_range(update_map.reverse()))

    constraints = (
        isl.ScheduleConstraints.on_domain(domain)
        .set_validity(validity)
        .set_coincidence(coincidence)
        .set_proximity(coincidence)
    )

    return constraints


def detect_constraint_stride(ctx, c, pos):
    stride = isl.Val.one(ctx)

    if not c.is_equality():
        return stride
    if not c.involves_dims(isl.dim_type.set, pos, 1):
        return stride

    stride = isl.Val.zero(ctx)
    n_div = c.dim(isl.dim_type.div)

    for i in xrange(n_div):
        stride = stride.gcd(c.get_coefficient_val(isl.dim_type.div, i))

    m = stride.gcd(c.get_coefficient_val(isl.dim_type.set, pos))
    stride = stride.div(m)

    if stride.is_zero():
        stride = isl.Val.one(ctx)
    return stride


def detect_stride(ctx, constraints, pos):
    stride = isl.Val.one(ctx)

    for c in constraints:
        s = detect_constraint_stride(ctx, c, pos)
        stride = stride.mul(s).div(stride.gcd(s))

    return stride.get_num_si()


def scale_down_node(node):
    domain = isl.Set.from_union_set(
        node.band_get_partial_schedule_union_map()
        .intersect_domain(node.get_domain())
        .range()).affine_hull()

    constraints = domain.get_constraints()
    n = domain.dim(isl.dim_type.set)

    ctx = node.get_ctx()

    strides = [detect_stride(ctx, constraints, i)
               for i in xrange(n)]

    mv = list_to_multival(node.band_get_space(), strides)
    node = node.band_scale_down(mv)
    return node


def mark_kernel(names, node):
    if node.get_type() == isl.schedule_node_type.band:
        node = scale_down_node(node)
        return node.insert_mark(isl.Id.alloc(node.get_ctx(), names.next(), None))

    n = node.n_children()
    for i in xrange(n):
        node = node.child(i)
        node = mark_kernel(names, node)
        node = node.parent()

    return node


def mark_kernels(schedule):
    return mark_kernel(Counter("kernel[%d]"), schedule.get_root()).get_schedule()
