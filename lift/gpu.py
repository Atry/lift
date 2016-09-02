# mostly stolen from ppcg

import islpy as isl
from .utils import product
from .isl_utils import find_kernel_id, get_sizes, list_to_multival, to_sets, to_maps, detect_strides


class KernelInfo(object):
    required_fields = ('domain', 'grid_sizes', 'block_sizes')

    def __init__(self, **kwargs):
        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self):
        return repr(self.__dict__)


def n_outer_coincidence(node):
    assert node.get_type() == isl.schedule_node_type.band
    assert node.band_get_permutable()

    n = node.band_n_member()

    for i in xrange(n):
        if not node.band_member_get_coincident(i):
            return i

    return n

def parameter_vector(ctx, prefix, n):
    space = isl.Space.set_alloc(ctx, n, n)
    for i in xrange(n):
        space = space.set_dim_name(isl.dim_type.param, i, prefix+str(i))

    bs = isl.BasicSet.universe(space)

    for i in xrange(n):
        bs = bs.add_constraint(
            isl.Constraint.alloc_equality(space)
            .set_coefficient_val(
                isl.dim_type.param, i,  isl.Val.int_from_si(ctx, 1)
            )
            .set_coefficient_val(
                isl.dim_type.set, i,  isl.Val.int_from_si(ctx, -1)
            ))

    return isl.Set.from_basic_set(bs)


def extract_sizes(set):
    n = set.dim(isl.dim_type.param)
    ls = isl.LocalSpace.from_space(set.get_space())
    return [
        set.max_val(
            isl.Aff.var_on_domain(ls, isl.dim_type.param, i))
        .get_num_si() + 1
        for i in xrange(n)]


def insert_context(node, grid_sizes, block_sizes):
    ctx = node.get_ctx()
    n_grid = len(grid_sizes)
    n_block = len(block_sizes)

    space = isl.Space.set_alloc(ctx, n_grid + n_block, 0)

    for i in xrange(n_grid):
        space = space.set_dim_name(isl.dim_type.param, i, "b"+str(i))

    for i in xrange(n_block):
        space = space.set_dim_name(isl.dim_type.param, n_grid+i, "t"+str(i))

    bs = isl.BasicSet.universe(space)

    for i in xrange(n_grid):
        bs = bs.add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, i,  isl.Val.int_from_si(ctx, -1)
            )
            .set_constant_val(
                isl.Val.int_from_si(ctx, grid_sizes[i] - 1)
            )
        ).add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, i,  isl.Val.int_from_si(ctx, 1)
            )
        )

    for i in xrange(n_block):
        bs = bs.add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, n_grid+i,  isl.Val.int_from_si(ctx, -1)
            )
            .set_constant_val(
                isl.Val.int_from_si(ctx, block_sizes[i] - 1)
            )
        ).add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.param, n_grid+i,  isl.Val.int_from_si(ctx, 1)
            )
        )

    return node.insert_context(isl.Set.from_basic_set(bs))


def set_schedule_modulo(node, prefix, sizes):
    mv = list_to_multival(node.band_get_space(), sizes)

    mupa = node.band_get_partial_schedule()
    mupa = mupa.mod_multi_val(mv)
    umap = isl.UnionMap.from_multi_union_pw_aff(mupa).reverse()

    return parameter_vector(node.get_ctx(), prefix, len(sizes)).apply(umap)


def scale_band(node, sizes):
    mv = list_to_multival(node.band_get_space(), sizes)
    node = node.band_scale(mv)
    return node


def is_marked(node, name):
    if node.get_type() != isl.schedule_node_type.mark:
        return False

    mark = node.mark_get_id()
    return mark.get_name() == name


def node_is_core(node, core):
    filter = node.filter_get_filter()
    return not filter.is_disjoint(core)


def core_child(node, core):
    if node.get_type() != isl.schedule_node_type.sequence:
        return node.child(0)

    n = node.n_children()
    for i in xrange(n):
        node = node.child(i)
        if node_is_core(node, core):
            return node.child(0)
        nod = node.parent()

    assert False


def gpu_tree_move_down_to(node, core, mark):
    while not is_marked(node, mark):
        node = core_child(node, core)

    return node


def gpu_tree_move_up_to(node, mark):
    while not is_marked(node, mark):
        node = node.parent()
    return node

def lower_bounds(set):
    n = set.dim(isl.dim_type.set)
    ls = isl.LocalSpace.from_space(set.get_space())
    return [
        set.min_val(
            isl.Aff.var_on_domain(ls, isl.dim_type.set, i))
        .get_num_si()
        for i in xrange(n)]

def upper_bounds(set):
    n = set.dim(isl.dim_type.set)
    ls = isl.LocalSpace.from_space(set.get_space())
    return [
        set.max_val(
            isl.Aff.var_on_domain(ls, isl.dim_type.set, i))
        .get_num_si() + 1
        for i in xrange(n)]


def schedule_copy_private(domain):
    space = domain.get_space().reset_tuple_id(isl.dim_type.set)
    mupa = isl.MultiUnionPwAff.from_union_map(domain.identity().reset_tuple_id(isl.dim_type.out))

    mv = list_to_multival(space, lower_bounds(domain))
    mupa2 = isl.MultiUnionPwAff.multi_val_on_domain(domain, mv)
    mupa = mupa.sub(mupa2)

    strides = detect_strides(domain.affine_hull())
    mv = list_to_multival(space, strides)

    node = isl.ScheduleNode.from_domain(domain)
    node = node.child(0)
    node = node.insert_partial_schedule(mupa)
    node = node.band_scale_down(mv)

    m = isl.Map.from_union_map(node.band_get_partial_schedule_union_map())
    return m


def map_to_scalar(ctx, sizes, param=False):
    n = len(sizes)
    if param:
        space = isl.Space.alloc(ctx, n, 1, 1)
    else:
        space = isl.Space.alloc(ctx, 0, n, 1)
    map = isl.Map.universe(space)

    t = isl.dim_type.param if param else isl.dim_type.in_

    for i in xrange(n):
        map = map.add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                t, i,  isl.Val.int_from_si(ctx, -1)
            )
            .set_constant_val(
                isl.Val.int_from_si(ctx, sizes[i] - 1)
            )
        ).add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                t, i,  isl.Val.int_from_si(ctx, 1)
            )
        )

    if param:
        map = map.add_constraint(
            isl.Constraint.alloc_inequality(space)
            .set_coefficient_val(
                isl.dim_type.in_, 0,  isl.Val.int_from_si(ctx, 1)
            )
        )

    c = isl.Constraint.alloc_equality(map.space)

    for i in xrange(n):
        c = c.set_coefficient_val(
            t, i,
            isl.Val.int_from_si(ctx, product(sizes[1+i:]))
        )
    c = c.set_coefficient_val(
        isl.dim_type.out, 0, isl.Val.int_from_si(ctx, -1))

    if param:
        c = c.set_coefficient_val(
            isl.dim_type.in_, 0,
            isl.Val.int_from_si(ctx, product(sizes))
        )

    map = map.add_constraint(c)

    if param:
        map = map.insert_dims(isl.dim_type.in_, 0, n)
        space = map.get_space()

        for i in xrange(n):
            map = map.add_constraint(
                isl.Constraint.alloc_equality(space)
                .set_coefficient_val(
                    isl.dim_type.param, i,  isl.Val.int_from_si(ctx, -1)
                )
                .set_coefficient_val(
                    isl.dim_type.in_, i,  isl.Val.int_from_si(ctx, 1)
                )
            )

    return map


def schedule_copy_local(domain, block_sizes):
    m = schedule_copy_private(domain)

    m1 = map_to_scalar(domain.get_ctx(), upper_bounds(m.range()))
    m2 = map_to_scalar(domain.get_ctx(), block_sizes, True)

    for i in xrange(len(block_sizes)):
        m2 = m2.set_dim_name(isl.dim_type.param, i, "t"+str(i))

    m = m.apply_range(m1)

    copy_schedule = m.apply_range(m2.reverse())
    storage_map = m.apply_range(m2.remove_dims(isl.dim_type.param, 0, 2).reverse())

    return copy_schedule, storage_map.set_tuple_name(isl.dim_type.out, "local_"+domain.get_tuple_name())



def create_kernel(ctx, node, grid_sizes, block_sizes):
    domain = node.get_domain()

    init_map = ctx.init_stmts.get_assign_map().intersect_domain(domain)
    update_map = ctx.update_stmts.get_assign_map().intersect_domain(domain)
    fini_map  = ctx.fini_stmts.get_assign_map().intersect_domain(domain)
    def_map = ctx.def_stmts.get_assign_map().intersect_domain(domain)
    use_map = ctx.get_use_map().intersect_domain(domain)
    assign_map = init_map.union(update_map).union(fini_map).union(def_map)

    writes = assign_map.range()
    reads = use_map.range().union(update_map.range().union(fini_map.range()).subtract(init_map.range()))

    accesses = writes.union(reads)
    access_map = assign_map.union(use_map)


    core = domain.universe()

    node = node.insert_mark(isl.Id.alloc(node.get_ctx(), "kernel", None))
    node = node.child(0)
    node = node.child(0)

    node = node.insert_mark(isl.Id.alloc(node.get_ctx(), "thread", None))
    node = node.parent()

    assert n_outer_coincidence(node) >= len(grid_sizes)

    if len(grid_sizes) < node.band_n_member():
        node = node.band_split(len(grid_sizes))

    block_filter = set_schedule_modulo(node, "b", grid_sizes)
    grid_sizes = extract_sizes(block_filter.intersect(node.get_domain()).params())

    node = scale_band(node, grid_sizes)

    node = gpu_tree_move_down_to(node, core, "thread")
    node = node.child(0)

    assert n_outer_coincidence(node) >= len(block_sizes)

    if len(block_sizes) < node.band_n_member():
        node = node.band_split(len(block_sizes))

    thread_filter = set_schedule_modulo(node, "t", block_sizes)
    block_sizes = extract_sizes(thread_filter.intersect(node.get_domain()).params())

    node = gpu_tree_move_up_to(node, "kernel")
    node = node.child(0)
    node = node.child(0)

    privates = [
        m.get_tuple_name(isl.dim_type.in_)
        for m in to_maps(access_map.apply_domain(node.get_prefix_schedule_union_map()).reverse())
        if m.is_single_valued() ]

    node = node.parent()
    node = insert_context(node, grid_sizes, block_sizes)
    node = node.child(0)

    node = node.insert_filter(block_filter)
    node = node.child(0)

    local_arrays = {
        s.get_tuple_name(): schedule_copy_local(s, block_sizes)
        for s in to_sets(access_map.apply_domain(
                node.get_prefix_schedule_union_map()).range())
        if s.get_tuple_name() not in privates}

    local_array_shapes = {k: upper_bounds(m.range()) for k,(m,_) in local_arrays.iteritems() }

    node = gpu_tree_move_down_to(node, core, "thread")
    node = node.delete()
    node = node.insert_filter(thread_filter)
    node = node.child(0)

    private_arrays = {
        s.get_tuple_name(): schedule_copy_private(s)
        for s in to_sets(access_map.apply_domain(
                node.get_prefix_schedule_union_map()).range())
        if s.get_tuple_name() in privates}
    private_array_shapes = {k: upper_bounds(m.range()) for k,m in private_arrays.iteritems() }
    private_arrays = {
        k: (m,m.set_tuple_name(isl.dim_type.out, "private_"+k))
        for k,m in private_arrays.iteritems()}

    node = gpu_tree_move_up_to(node, "kernel")
    node = node.child(0)
    node = node.child(0)
    node = node.child(0)
    node = node.child(0)

    node = gpu_tree_move_up_to(node, "kernel")
    node = node.delete()
    node = node.insert_mark(node.parent().mark_get_id())

    info = KernelInfo(
        domain=domain,
        grid_sizes=grid_sizes,
        block_sizes=block_sizes,
        reads = list({ s.get_tuple_name() for s in to_sets(reads) }),
        writes = list({ s.get_tuple_name() for s in to_sets(writes) }),
        arrays = list({ s.get_tuple_name() for s in to_sets(accesses) }),
        privates = privates,
        local_arrays = local_arrays,
        local_array_shapes = local_array_shapes,
        private_arrays = private_arrays,
        private_array_shapes = private_array_shapes)

    return node, info


def is_permutable(node):
    node_type = node.get_type()

    if node_type != isl.schedule_node_type.band:
        return False
    if not node.band_get_permutable():
        return False
    if node.band_n_member() < 1:
        return False
    if not node.band_member_get_coincident(0):
        return False
    return True


def create_kernels(ctx, kernels, node, sizes):
    kernel_id = find_kernel_id(node)
    if kernel_id is not None:
        node = node.child(0)
        if is_permutable(node):
            grid_sizes = get_sizes(sizes, "grid", kernel_id)
            block_sizes = get_sizes(sizes, "block", kernel_id)
            if grid_sizes is not None and block_sizes is not None:
                assert len(grid_sizes) == len(block_sizes)
                node, info = create_kernel(ctx, node, grid_sizes, block_sizes)
                kernels[kernel_id] = info

        node = node.parent()
        node = node.delete()
        return node

    n = node.n_children()

    for i in xrange(n):
        node = node.child(i)
        node = create_kernels(ctx, kernels, node, sizes)
        node = node.parent()

    return node


def map_to_device(ctx, schedule, sizes):
    node = schedule.get_root()
    kernels = {}
    node = create_kernels(ctx, kernels, node, sizes)
    return node.get_schedule(), kernels
