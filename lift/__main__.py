import islpy as isl

from .parser import Parser
from .check import check_stmts
from .compile import compile
from .contract import contract_arrays
from .codegen import get_schedule_constraints, mark_kernels, codegen
from .tile import tile_kernels
from .gpu import map_to_device
from .opencl_formatter import format_opencl

import sys

basename = sys.argv[1]
name = sys.argv[2]
stmts = ()

for filename in sys.argv[3:]:
    p = Parser(filename=filename)

    with open(filename, "r") as f:
        source = f.read()

    stmts += p.parse(source)

table = check_stmts(stmts)
context = compile(table)
contract_arrays(context)

constraints = get_schedule_constraints(context)

context.isl_context.set_ast_build_detect_min_max(1)
context.isl_context.set_schedule_maximize_band_depth(1)
context.isl_context.set_schedule_maximize_coincidence(1)
context.isl_context.set_schedule_whole_component(0)
context.isl_context.set_schedule_separate_components(1)
context.isl_context.set_schedule_treat_coalescing(1)
context.isl_context.set_schedule_outer_coincidence(1)

sizes = isl.UnionMap(
    "{ kernel[i] -> grid[2,2]; kernel[i] -> block[2,2]; kernel[i] -> tile[2,2]}",
    context.isl_context)

schedule = constraints.compute_schedule()
schedule = mark_kernels(schedule)
schedule = tile_kernels(schedule, sizes)
schedule, kernels = map_to_device(context, schedule, sizes)
ast = codegen(context, schedule, kernels)
print format_opencl(name, table, context, ast, kernels)
