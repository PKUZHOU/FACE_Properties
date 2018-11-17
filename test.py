import torch
from network import Property_Net
import cv2
import os
from torch.autograd.variable import Variable
from mtcnn_pytorch.src.detector import detect_faces
from PIL import Image
import torch.nn.functional as F

def detect(path):
    '''
    :param path: imagefile path
    :return: detected bounding_boxes
    '''
    image = Image.open(path)
    bounding_boxes, landmarks = detect_faces(image)
    return bounding_boxes

def detect_property(net,path):
        img = cv2.imread(path)
        resized = cv2.resize(img,(32,32))
        resized = torch.Tensor(resized).permute(2,0,1).unsqueeze(0)/255.
        resized = Variable(resized).cuda()
        glass_pred, mask_pred = net(resized)#*45
        glass_prob = F.softmax(glass_pred, dim=1)[0][1]
        mask_prob = F.softmax(mask_pred, dim =1)[0][1]
        print "glass prob",glass_prob
        print "mask prob", mask_prob
        # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
        # cv2.putText(img,str(float(pose[0].cpu().data[0]))[:3],(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,255),fontScale=0.6,thickness=1)
        # cv2.imwrite('test.jpg',img)

def test():
    net = PoseNet(4)
    net.train(False)
    net.cuda()
    net.load_state_dict(torch.load('./MK_models/model_29.pkl'))
    root = "./images"
    imgs = os.listdir(root)
    for img in imgs:
        path = os.path.join(root,img)
        boxes = detect(path)
        detect_pose(net,path,boxes)

if __name__ == '__main__':
    # test()
    net = Property_Net()
    net.cuda()
    detect_property(net,'/home/zhou/Pictures/timg.jpeg')
