import tvm
def test_1():
	m=64
	n=128
	shape=(64,128)
	A=tvm.placeholder(shape,name="A")
	C=tvm.compute(shape,lambda *indice : tvm.log(A(*indice)),name="C")
	s=tvm.create_schedule(C.op)
	print(tvm.lower(s,[A,C],simple_mode=True))

	pass

def test_voxel_expr():

	pass


if __name__ == '__main__':
	test_1()