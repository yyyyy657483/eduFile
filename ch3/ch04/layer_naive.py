"""
## 오차 역전파법  
MulLayer(곱셈노드)와 AddLayer(덧셈노드)를 구현해보자
"""

# coding: utf-8
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 곱셈의 순전파
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    # 곱셈의 역전파
    def backward(self, dout):
        dx = dout * self.y # x와 y를 바꾼다
        dy = dout * self.x

        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass    

    # 덧셈의 순전파
    def forward(self, x, y):
        out = x + y
        return out
    
    # 덧셈의 역전파
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
        