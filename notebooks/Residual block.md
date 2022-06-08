flowchart TB
	title[<u>Residual block</u>]
style title fill:#FFF,stroke:#FFF
	x(x)
	blocks_0_conv("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	x -->blocks_0_conv 
	blocks_0_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	blocks_0_conv -->blocks_0_bn 
	blocks_0_act("ReLU[inplace=True]")
	blocks_0_bn -->blocks_0_act 
	blocks_1_conv("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	blocks_0_act -->blocks_1_conv 
	blocks_1_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	blocks_1_conv -->blocks_1_bn 
	blocks_1_act("ReLU[inplace=True]")
	blocks_1_bn -->blocks_1_act 
	add{add}
	x  & blocks_1_act -..->add 
	activation("ReLU[inplace=True]")
	add -->activation 
	output>output]
	activation -->output 
