from .parser import Parser
from .check import check_stmts
from .compile import compile
from .contract import contract_arrays
from .codegen import codegen
from .asmjs_formatter import format_asmjs

import sys

name = sys.argv[1]
stmts = ()

for filename in sys.argv[2:]:
    p = Parser(filename=filename)

    with open(filename, "r") as f:
        source = f.read()

    stmts += p.parse(source)

table = check_stmts(stmts)
context = compile(table)
contract_arrays(context)
context.isl_context.set_schedule_serialize_sccs(1)
ast = codegen(context)
print format_asmjs(name, table, context, ast)
