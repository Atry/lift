import re
import islpy as isl

isl.schedule_node_type = isl._isl.schedule_node_type

def isl_schedule_node_get_type(node):
    return isl._isl.lib.isl_schedule_node_get_type(node.data)

isl.ScheduleNode.get_type = isl_schedule_node_get_type

def isl_constraint_dim(c, type):
    return isl._isl.lib.isl_constraint_dim(c.data, type)

isl.Constraint.dim = isl_constraint_dim


def to_list(l):
    r = []
    l.foreach(lambda e: r.append(e))
    return r


def to_sets(union_set):
    sets = []
    union_set.foreach_set(lambda x: sets.append(x))
    return sets


def to_maps(union_map):
    maps = []
    union_map.foreach_map(lambda x: maps.append(x))
    return maps


def list_to_multival(space, l):
    ctx = space.get_ctx()
    mv = isl.MultiVal.zero(space)
    for i, e in enumerate(l):
        v = isl.Val.int_from_si(ctx, e)
        mv = mv.set_val(i,v)
    return mv


def find_kernel_id(node):
    if node.get_type() == isl.schedule_node_type.mark:
        mark = node.mark_get_id().get_name()
        match = re.match(r'kernel\[(\d+)\]', mark)
        if match is not None:
            return int(match.group(1))


def get_sizes(sizes, name, kernel_id):
    ctx = sizes.get_ctx()
    space = isl.Space.set_alloc(ctx, 0, 1)
    space = space.set_tuple_name(isl.dim_type.set, "kernel")
    uset = (
        isl.Set.universe(space)
        .add_constraint(
            isl.Constraint.alloc_equality(space)
            .set_coefficient_val(
                isl.dim_type.set, 0, isl.Val.int_from_si(ctx, 1))
            .set_constant_val(
                isl.Val.int_from_si(ctx, kernel_id)))
    )

    for set in to_sets(uset.apply(sizes)):
        if set.get_tuple_name() == name:
            n = set.dim(isl.dim_type.set)

            return [
                set.plain_get_val_if_fixed(
                    isl.dim_type.set, i).get_num_si()
                for i in xrange(n)]
