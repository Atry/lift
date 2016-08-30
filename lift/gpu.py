import islpy as isl
from .isl_utils import find_kernel_id, get_sizes, list_to_multival

class KernelInfo(object):
    required_fields = ('domain', 'grid_sizes', 'block_sizes')

    def __init__(self, **kwargs):
        assert all((k in kwargs) for k in self.required_fields)

        for k,v in kwargs.items():
            self.__dict__[k] = v

    def __repr__(self):
        return repr(self.__dict__)


def n_outer_coincidence(node):
    assert isl_schedule_node_get_type(node) == schedule_node_type.band
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
    obj = isl.Aff.zero_on_domain(ls)
    return [
        set.max_val(
            obj.set_coefficient_val(
                isl.dim_type.param, 1,
                isl.Val.int_from_si(set.get_ctx(), 1)))
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
    ctx = node.get_ctx()
    space = node.band_get_space()
    mv = isl.MultiVal.zero(space)
    for i in xrange(len(sizes)):
        v = isl.Val.int_from_si(ctx, sizes[i])
        mv = mv.set_val(i,v)

    mupa = node.band_get_partial_schedule()
    mupa = mupa.mod_multi_val(mv)
    umap = isl.UnionMap.from_multi_union_pw_aff(mupa).reverse()

    return parameter_vector(ctx, prefix, len(sizes)).apply(umap)


def scale_band(node, sizes):
    mv = list_to_multival(node.band_get_space(), sizes)
    node = node.band_scale(mv)
    return node


def create_kernel(ctx, node, grid_sizes, block_sizes):
    domain = node.get_domain()
    block_filter = set_schedule_modulo(node, "b", grid_sizes)
    thread_filter = set_schedule_modulo(node.child(0), "t", block_sizes)

    grid_sizes = extract_sizes(block_filter.intersect(node.get_domain()).params())
    block_sizes = extract_sizes(thread_filter.intersect(node.child(0).get_domain()).params())


    node = insert_context(node, grid_sizes, block_sizes)
    node = node.child(0)

    node = node.child(0)

    if len(block_sizes) < node.band_n_member():
        node = node.band_split(len(block_sizes))

    node = node.insert_filter(thread_filter)
    node = node.parent()


    if len(grid_sizes) < node.band_n_member():
        node = node.band_split(len(grid_sizes))

    node = scale_band(node, grid_sizes)

    node = node.insert_filter(block_filter)
    node = node.parent()

    node = node.insert_mark(node.parent().mark_get_id())

    info = KernelInfo(
        domain=domain,
        grid_sizes=grid_sizes,
        block_sizes=block_sizes)

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
