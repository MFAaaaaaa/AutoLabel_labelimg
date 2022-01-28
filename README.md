# AutoLabel_labelimg
A semi-automatic labeling tool for image labels;利用opecncv中支持darkne框架的接口，实现对图像标签的半自动标注，可利用labelimg人工校准。
# Windows
1、安装需要的第三方类库

2、修改autoLabel.py中的权重路径、待标注的图像文件夹路径、标注完成后的存储路径

weightsPath = "./weights/bingxian.weights"

configPath = "./cfg/bingxian.cfg"

labelsPath = "./names/bingxian.names"

imgdir = "E:/Data/2021-10-15/JPEGS"  # 待标注的图片

tree.write('E:/Data/2021-10-15/Annotations/{0}.xml'.format(a)) # xml文件存储路径

3、标注完成后可利用labelConfirm.py检查标注效果
