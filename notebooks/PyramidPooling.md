flowchart TB
	title[<u>PyramidPooling</u>]
style title fill:#FFF,stroke:#FFF
	x(x)
	getitem{getitem}
	x -..->getitem 
	sizesize
	features_0_0("AdaptiveAvgPool2d[output_size=[0, 1]]")
	x -->features_0_0 
	features_0_1_conv("Conv2d[254, 64, kernel_size=[1, 1], stride=[1, 1], padding=[1, 1], bias=False]")
	features_0_0 -->features_0_1_conv 
	features_0_1_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	features_0_1_conv -->features_0_1_bn 
	features_0_1_act("ReLU[inplace=True]")
	features_0_1_bn -->features_0_1_act 
	interpolate{interpolate}
	features_0_1_act  & size -..->interpolate 
	features_1_0("AdaptiveAvgPool2d[output_size=[1, 2]]")
	x -->features_1_0 
	features_1_1_conv("Conv2d[254, 64, kernel_size=[1, 1], stride=[1, 1], padding=[1, 1], bias=False]")
	features_1_0 -->features_1_1_conv 
	features_1_1_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	features_1_1_conv -->features_1_1_bn 
	features_1_1_act("ReLU[inplace=True]")
	features_1_1_bn -->features_1_1_act 
	interpolate_1{interpolate_1}
	features_1_1_act  & size -..->interpolate_1 
	features_2_0("AdaptiveAvgPool2d[output_size=[2, 3]]")
	x -->features_2_0 
	features_2_1_conv("Conv2d[254, 64, kernel_size=[1, 1], stride=[1, 1], padding=[1, 1], bias=False]")
	features_2_0 -->features_2_1_conv 
	features_2_1_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	features_2_1_conv -->features_2_1_bn 
	features_2_1_act("ReLU[inplace=True]")
	features_2_1_bn -->features_2_1_act 
	interpolate_2{interpolate_2}
	features_2_1_act  & size -..->interpolate_2 
	features_3_0("AdaptiveAvgPool2d[output_size=[3, 5]]")
	x -->features_3_0 
	features_3_1_conv("Conv2d[254, 64, kernel_size=[1, 1], stride=[1, 1], padding=[1, 1], bias=False]")
	features_3_0 -->features_3_1_conv 
	features_3_1_bn("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	features_3_1_conv -->features_3_1_bn 
	features_3_1_act("ReLU[inplace=True]")
	features_3_1_bn -->features_3_1_act 
	interpolate_3{interpolate_3}
	features_3_1_act  & size -..->interpolate_3 
	cat{cat}
	x  & interpolate  & interpolate_1  & interpolate_2  & interpolate_3 -..->cat 
	output>output]
