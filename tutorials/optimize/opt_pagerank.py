""" How to optimize pagerank on tvm and cblas
==========================================
**Author**: `Hui Li <li7hui@gmail.com>`_

PageRank is the well know link analysis algorithm. It calculates numerical value to each element of a hyperlinked set of web pages, which reflects the probability that the random surfer will access that page. The process of PageRank can be understood as a Markov chain which needs recursively calculation to converge. So it is within the class of applications where sgemm_ and multiple iterations are necessary for the overall computation.

This work of PageRank refers to this paper:
"Twister: A Runtime for Iterative MapReduce,"  Jaliya Ekanayake, Hui Li
"""
##########################################
# Preparation and Algorithm
# -------------------------
"""
In PageRank, the goal is to calculate the access probability for each web page. An iteration of the algorithm calculates the new access probability for each web page based on values calculated in the previous computation. The iteration will not stop until the difference value is less than a predefined threshold, where the difference value is the different between the access probabilities of web pages in (N)th iteration and the those in (N+1)th iteration. 
"""
import tvm
import numpy
import time
from tvm.contrib import cblas

# The size of the web graph
N = 100

page_links        = tvm.nd.array(numpy.random.randint(2,size=(N,N)).astype("float32"),tvm.cpu(0))
page_rank_initial = tvm.nd.array(numpy.random.rand(N,1).astype("float32"),tvm.cpu(0))
page_rank_update  = tvm.nd.array(numpy.random.rand(N,1).astype("float32"),tvm.cpu(0))

# Algorithm
PR 		= tvm.placeholder((N,1),name = 'PageRank')
Graph		= tvm.placeholder((N,N),name = 'WebGraph')
k 		= tvm.reduce_axis((0,N),'k')

OutDegrees = tvm.compute(PR.shape,
		lambda x,_:tvm.sum(Graph[x,k],axis=k),
		name = 'OutDegrees')

Weights = tvm.compute(Graph.shape,
		lambda x,y : Graph[x][y]/(1+ OutDegrees[x][0]),
		name = 'Weights')

k = tvm.reduce_axis((0,N),'k')
PR_UPDATE = tvm.compute(PR.shape,
		lambda x,_: tvm.sum(PR[k][0]*Weights[k,x],axis=k),
		name = 'PR_UPDATE')

s = tvm.create_schedule(PR_UPDATE.op)

func = tvm.build(s,[PR,Graph,PR_UPDATE], name = 'pagerank')
assert func
evaluator = func.time_evaluator(func.entry_name,tvm.cpu(0),number = 1)

T = 0.0001
while True:

	#calculate the new pagerank
	evaluator(page_rank_initial,page_links,page_rank_update)

	#calculate the stop condition
	diff = 0.0
	for pr_old,pr_update in zip(page_rank_initial.asnumpy(),page_rank_update.asnumpy()):
		diff = numpy.max([diff,numpy.abs(pr_old-pr_update)])	
	print(diff)
	if diff<T:
		break
	#update page_rank_initial with page_rank_update	
	page_rank_initial = tvm.nd.array(page_rank_update.asnumpy().astype("float32"),tvm.cpu(0))

#################################
# PageRank with cblas
# apply tvm.contrib.cblas.matmul to core computation
# Algorithm:
#     Weights   =[N][N]
#     PR        =[N][1]
#     PR_Update = matmul(W,PR)

from tvm.contrib import cblas

N = 100
page_links        = tvm.nd.array(numpy.random.randint(2,size=(N,N)).astype("float32"),tvm.cpu(0))
page_rank_initial = tvm.nd.array(numpy.random.rand(N,1).astype("float32"),tvm.cpu(0))
page_rank_update  = tvm.nd.array(numpy.random.rand(N,1).astype("float32"),tvm.cpu(0))

#store the transposed the weights matrix first in order to apply matmul 
page_weights_transposed  = tvm.nd.array(numpy.random.rand(N,N).astype("float32"),tvm.cpu(0))

PR              = tvm.placeholder((N,1),name = 'PageRank')
Graph           = tvm.placeholder((N,N),name = 'WebGraph')
k               = tvm.reduce_axis((0,N),'k')

OutDegrees = tvm.compute(PR.shape,
                lambda x,_:tvm.sum(Graph[x,k],axis=k),
                name = 'OutDegrees')

#transpos the weights matrix
Weights_Transposed 	= tvm.compute(Graph.shape,
                	lambda x,y : Graph[y][x]/(1+ OutDegrees[y][0]),
                	name = 'Weights')

s = tvm.create_schedule(Weights_Transposed.op)
func = tvm.build(s,[Graph,Weights_Transposed], name = 'weights_transposed')
assert func
evaluator = func.time_evaluator(func.entry_name,tvm.cpu(0),number = 1)
evaluator(page_links,page_weights_transposed)

Weights       = tvm.placeholder((N,N),name = 'Weights')

#apply sgemm in cblas
PR_UPDATE 	=  cblas.matmul(Weights,PR)

s = tvm.create_schedule(PR_UPDATE.op)
func = tvm.build(s,[PR,Weights,PR_UPDATE], name = 'pagerank')
assert func
evaluator = func.time_evaluator(func.entry_name,tvm.cpu(0),number = 1)

print("PageRank with cblas")
T = 0.0001
while True:

        #calculate the new pagerank
        evaluator(page_rank_initial,page_weights_transposed,page_rank_update)

        #calculate the stop condition
        diff = 0.0
        for pr_old,pr_update in zip(page_rank_initial.asnumpy(),page_rank_update.asnumpy()):
                diff = numpy.max([diff,numpy.abs(pr_old-pr_update)])
        print(diff)
        if diff<T:
                break
        #update page_rank_initial with page_rank_update
        page_rank_initial = tvm.nd.array(page_rank_update.asnumpy().astype("float32"),tvm.cpu(0))

