import torch.cuda
import torchvision.transforms as transforms
from model import MobileNetV3
from PIL import Image


class Detector(object):
    def __init__(self, net_kind, num_classes=17):
        super(Detector, self).__init__()
        kind = net_kind.lower()
        if kind == 'large':
            self.net = MobileNetV3.MobileNetV3_Large(num_classes=num_classes)
        elif kind == 'small':
            self.net = MobileNetV3.MobileNetV3_Small(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self, weight_path):
        self.net.load_state_dict(torch.load(weight_path))

    def detect(self, weight_path, pic_path):
        self.load_weights(weight_path=weight_path)
        img = Image.open(pic_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        net_output = self.net(img_tensor)
        print(net_output)
        _, predicted = torch.max(net_output.data, 1)
        result = predicted[0].item()
        print("预测的结果为：", result)


detector = Detector('large', num_classes=17)
detector.detect('./weights/best.pkl', './1.jpg')
