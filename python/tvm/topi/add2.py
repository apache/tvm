# from tvm.te import hybrid

# @hybrid.script
# def add_2_helper(data1, data2):
#     out = output_tensor(data1.shape, data1.dtype)
#     out = data1 + data2
#     return


# def add2(data1, data2):
#     """Update data by adding values in updates at positions defined by indices

#     Parameters
#     ----------
#     data : relay.Expr
#         The input data to the operator.

#     indices : relay.Expr
#         The index locations to update.

#     updates : relay.Expr
#         The values to update.

#     axis : int
#         The axis to scatter_add on

#     Returns
#     -------
#     ret : relay.Expr
#         The computed result.
#     """
#     return data1 + data2
