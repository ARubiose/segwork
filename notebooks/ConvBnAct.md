flowchart TB
	title[<u>ConvBnAct</u>]
style title fill:#FFF,stroke:#FFF
	x(x)
	conv("Conv2d[254, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	x -->conv 
	bn("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	conv -->bn 
	act("ReLU[inplace=True]")
	bn -->act 
	output>output]
	act -->output 
