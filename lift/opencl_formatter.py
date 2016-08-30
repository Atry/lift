from .c_formatter import format_ast, format_arguments

OPENCL_TEMPLATE = """__kernel
void
kernel{id}(
{arguments}){{
{iterators}
{body}
}}
"""

def format_kernel_iterators(kernel):
    for i in xrange(len(kernel.grid_sizes)):
        yield "int b{0} = get_group_id({0});\n".format(i)

    for i in xrange(len(kernel.block_sizes)):
        yield "int t{0} = get_local_id({0});\n".format(i)


def format_kernel_arguments(table, kernel):
    for v in kernel.arrays:
        yield "__global float %s%s"%(v, "".join("[%d]"%(s,) for s in table.vars[v].shape[::-1]))


def format_kernel(table, kernels):
    for kernel_id, kernel in kernels.iteritems():
        yield OPENCL_TEMPLATE.format(
            id = kernel_id,
            arguments = ",\n".join(format_kernel_arguments(table, kernel)),
            iterators = "".join(format_kernel_iterators(kernel)),
            body = format_ast(table, kernel.ast)
        )

C_TEMPLATE = """#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#define inf (1.0/0.0)
#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

static const char kernel[] = "{kernel}";

void
{name}(
cl_device_id device,
{arguments}){{

cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
size_t size = sizeof(kernel);
cl_program program = clCreateProgramWithSource(context, 1, &kernel, &size, NULL);
if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS){{
  size_t len;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char log[len];
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
  fprintf(stderr, "%s\\n", log);
  clReleaseProgram(program);
  clReleaseContext(context);
  return;
}}

cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

{body}

clReleaseCommandQueue(queue);
clReleaseProgram(program);
clReleaseContext(context);
}}
"""


KERNEL_TEMPLATE = """{{
cl_kernel kernel = clCreateKernel(program, "kernel{id}", NULL);
{arguments}
size_t global_sizes[] = {{ {global_sizes} }};
size_t block_sizes[] = {{ {block_sizes} }};
clEnqueueNDRangeKernel(queue, kernel, {n}, NULL, global_sizes, block_sizes, 0, NULL, NULL);
clReleaseKernel(kernel);
}}
"""

def format_opencl(name, table, ctx, ast, kernels):
    def format_opencl_kernel(kernel_id):
        kernel = kernels[kernel_id]
        return KERNEL_TEMPLATE.format(
            id = kernel_id,
            global_sizes = ", ".join(
                "%d*%d"%(a,b)
                for a,b in zip(kernel.grid_sizes,kernel.block_sizes)),
            block_sizes = ", ".join(str(b) for b in kernel.block_sizes),
            arguments = "".join(
                "clSetKernelArg(kernel, {}, sizeof(cl_mem), (void *)&mem_{});\n".format(i,v)
                for i, v in enumerate(kernel.arrays)),
            n = len(kernel.grid_sizes)
        )

    return C_TEMPLATE.format(
        name = name,
        arguments = ",\n".join(format_arguments(table, ctx)),
        kernel = repr("\n".join(format_kernel(table, kernels)))[1:-1],
        body = format_ast(table, ast, format_opencl_kernel))
