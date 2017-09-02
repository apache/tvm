#PageRank with TVM li7hui@gmail.com

import tvm
import numpy
import time

# The size of the web graph
N = 100

np_page_rank = numpy.random.rand(N,)
np_page_links = numpy.random.randint(2,size=(N,N))

page_rank_initial = tvm.nd.array(np_page_rank.astype("float32"),tvm.cpu(0))
page_links = tvm.nd.array(np_page_links.astype("float32"),tvm.cpu(0))
page_rank_update = tvm.nd.array(numpy.random.rand(N,).astype("float32"),tvm.cpu(0))

# Algorithm
PR = tvm.placeholder((N,),name = 'PageRank')
Graph = tvm.placeholder((N,N),name = 'WebGraph')
k = tvm.reduce_axis((0,N),'k')
OutDegrees = tvm.compute(PR.shape,
		lambda x:tvm.sum(Graph[x,k],axis=k),
		name = 'OutDegrees')
Weights = tvm.compute(Graph.shape,
		lambda x,y : Graph[x][y]/(1+ OutDegrees[x]),
		name = 'Weights')
k = tvm.reduce_axis((0,N),'k')
PR_UPDATE = tvm.compute(PR.shape,
		lambda x: tvm.sum(PR[k]*Weights[k,x],axis=k),
		name = 'PR_UPDATE')
s = tvm.create_schedule(PR_UPDATE.op)

func = tvm.build(s,[PR,Graph,PR_UPDATE], name = 'pagerank')
assert func
evaluator = func.time_evaluator(func.entry_name,tvm.cpu(0),number = 1)

iter = 0
while True:
	evaluator(page_rank_initial,page_links,page_rank_update)
	page_rank_initial = page_rank_update
	iter = iter + 1
	if iter>10:
		break

