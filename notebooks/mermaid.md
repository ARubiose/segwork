flowchart TB
	title[<u>timm/resnet18</u>]
    style title fill:#FFF,stroke:#FFF
	x(x)
	conv1("Conv2d[3, 64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], bias=False]")
	x -->conv1 
	bn1("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	conv1 -->bn1 
	act1("ReLU[inplace=True]")
	bn1 -->act1 
	maxpool("MaxPool2d[kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False]")
	act1 -->maxpool 
	layer1_0_conv1("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	maxpool -->layer1_0_conv1 
	layer1_0_bn1("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer1_0_conv1 -->layer1_0_bn1 
	layer1_0_act1("ReLU[inplace=True]")
	layer1_0_bn1 -->layer1_0_act1 
	layer1_0_conv2("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer1_0_act1 -->layer1_0_conv2 
	layer1_0_bn2("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer1_0_conv2 -->layer1_0_bn2 
	add{add}
	layer1_0_bn2  & maxpool -..->add 
	layer1_0_act2("ReLU[inplace=True]")
	add -->layer1_0_act2 
	layer1_1_conv1("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer1_0_act2 -->layer1_1_conv1 
	layer1_1_bn1("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer1_1_conv1 -->layer1_1_bn1 
	layer1_1_act1("ReLU[inplace=True]")
	layer1_1_bn1 -->layer1_1_act1 
	layer1_1_conv2("Conv2d[64, 64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer1_1_act1 -->layer1_1_conv2 
	layer1_1_bn2("BatchNorm2d[64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer1_1_conv2 -->layer1_1_bn2 
	add_1{add_1}
	layer1_1_bn2  & layer1_0_act2 -..->add_1 
	layer1_1_act2("ReLU[inplace=True]")
	add_1 -->layer1_1_act2 
	layer2_0_conv1("Conv2d[64, 128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False]")
	layer1_1_act2 -->layer2_0_conv1 
	layer2_0_bn1("BatchNorm2d[128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer2_0_conv1 -->layer2_0_bn1 
	layer2_0_act1("ReLU[inplace=True]")
	layer2_0_bn1 -->layer2_0_act1 
	layer2_0_conv2("Conv2d[128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer2_0_act1 -->layer2_0_conv2 
	layer2_0_bn2("BatchNorm2d[128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer2_0_conv2 -->layer2_0_bn2 
	layer2_0_downsample_0("Conv2d[64, 128, kernel_size=[1, 1], stride=[2, 2], bias=False]")
	layer1_1_act2 -->layer2_0_downsample_0 
	layer2_0_downsample_1("BatchNorm2d[128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer2_0_downsample_0 -->layer2_0_downsample_1 
	add_2{add_2}
	layer2_0_bn2  & layer2_0_downsample_1 -..->add_2 
	layer2_0_act2("ReLU[inplace=True]")
	add_2 -->layer2_0_act2 
	layer2_1_conv1("Conv2d[128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer2_0_act2 -->layer2_1_conv1 
	layer2_1_bn1("BatchNorm2d[128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer2_1_conv1 -->layer2_1_bn1 
	layer2_1_act1("ReLU[inplace=True]")
	layer2_1_bn1 -->layer2_1_act1 
	layer2_1_conv2("Conv2d[128, 128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer2_1_act1 -->layer2_1_conv2 
	layer2_1_bn2("BatchNorm2d[128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer2_1_conv2 -->layer2_1_bn2 
	add_3{add_3}
	layer2_1_bn2  & layer2_0_act2 -..->add_3 
	layer2_1_act2("ReLU[inplace=True]")
	add_3 -->layer2_1_act2 
	layer3_0_conv1("Conv2d[128, 256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False]")
	layer2_1_act2 -->layer3_0_conv1 
	layer3_0_bn1("BatchNorm2d[256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer3_0_conv1 -->layer3_0_bn1 
	layer3_0_act1("ReLU[inplace=True]")
	layer3_0_bn1 -->layer3_0_act1 
	layer3_0_conv2("Conv2d[256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer3_0_act1 -->layer3_0_conv2 
	layer3_0_bn2("BatchNorm2d[256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer3_0_conv2 -->layer3_0_bn2 
	layer3_0_downsample_0("Conv2d[128, 256, kernel_size=[1, 1], stride=[2, 2], bias=False]")
	layer2_1_act2 -->layer3_0_downsample_0 
	layer3_0_downsample_1("BatchNorm2d[256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer3_0_downsample_0 -->layer3_0_downsample_1 
	add_4{add_4}
	layer3_0_bn2  & layer3_0_downsample_1 -..->add_4 
	layer3_0_act2("ReLU[inplace=True]")
	add_4 -->layer3_0_act2 
	layer3_1_conv1("Conv2d[256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer3_0_act2 -->layer3_1_conv1 
	layer3_1_bn1("BatchNorm2d[256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer3_1_conv1 -->layer3_1_bn1 
	layer3_1_act1("ReLU[inplace=True]")
	layer3_1_bn1 -->layer3_1_act1 
	layer3_1_conv2("Conv2d[256, 256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer3_1_act1 -->layer3_1_conv2 
	layer3_1_bn2("BatchNorm2d[256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer3_1_conv2 -->layer3_1_bn2 
	add_5{add_5}
	layer3_1_bn2  & layer3_0_act2 -..->add_5 
	layer3_1_act2("ReLU[inplace=True]")
	add_5 -->layer3_1_act2 
	layer4_0_conv1("Conv2d[256, 512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], bias=False]")
	layer3_1_act2 -->layer4_0_conv1 
	layer4_0_bn1("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer4_0_conv1 -->layer4_0_bn1 
	layer4_0_act1("ReLU[inplace=True]")
	layer4_0_bn1 -->layer4_0_act1 
	layer4_0_conv2("Conv2d[512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer4_0_act1 -->layer4_0_conv2 
	layer4_0_bn2("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer4_0_conv2 -->layer4_0_bn2 
	layer4_0_downsample_0("Conv2d[256, 512, kernel_size=[1, 1], stride=[2, 2], bias=False]")
	layer3_1_act2 -->layer4_0_downsample_0 
	layer4_0_downsample_1("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer4_0_downsample_0 -->layer4_0_downsample_1 
	add_6{add_6}
	layer4_0_bn2  & layer4_0_downsample_1 -..->add_6 
	layer4_0_act2("ReLU[inplace=True]")
	add_6 -->layer4_0_act2 
	layer4_1_conv1("Conv2d[512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer4_0_act2 -->layer4_1_conv1 
	layer4_1_bn1("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer4_1_conv1 -->layer4_1_bn1 
	layer4_1_act1("ReLU[inplace=True]")
	layer4_1_bn1 -->layer4_1_act1 
	layer4_1_conv2("Conv2d[512, 512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], bias=False]")
	layer4_1_act1 -->layer4_1_conv2 
	layer4_1_bn2("BatchNorm2d[512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True]")
	layer4_1_conv2 -->layer4_1_bn2 
	add_7{add_7}
	layer4_1_bn2  & layer4_0_act2 -..->add_7 
	layer4_1_act2("ReLU[inplace=True]")
	add_7 -->layer4_1_act2 
	output>output]
	act1  & layer1_1_act2  & layer2_1_act2  & layer3_1_act2  & layer4_1_act2 -->output 
