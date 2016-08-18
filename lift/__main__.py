from .parser import Parser
from .check import check_stmts
from .compile import compile
from .contract import contract_arrays
from .codegen import get_schedule_constraints, codegen
from .c_formatter import format_c

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

context.isl_context.set_schedule_serialize_sccs(1)
ast = codegen(context, constraints)

with open(basename+".c", "w") as f:
    f.write(format_c(name, table, context, ast))

context.isl_context.set_schedule_serialize_sccs(0)
context.isl_context.set_ast_build_detect_min_max(1)
context.isl_context.set_schedule_maximize_band_depth(1)
context.isl_context.set_schedule_maximize_coincidence(1)
context.isl_context.set_schedule_whole_component(0)
context.isl_context.set_schedule_separate_components(1)

schedule = constraints.compute_schedule()

with open(basename+".yaml","w") as f:
    f.write(schedule.to_str())

