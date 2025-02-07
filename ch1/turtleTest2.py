"""
거북이 그리기


class name: littleTurtle
function:
    setting: 거북이 기본값 세팅한다
        self 
        x: 마우스 x 좌표
        y: 마우스 y 좌표
    randColor: 박스 컬러를 바꾼다
    drawRectangle: 박스를 그린다

main
function:
    createTurtle: 마우스를 클릭할 때마다 littleTurtle 객체를 새로 만든다
    parameter:
        x: 마우스 x 좌표
        y: 마우스 y 좌표

"""

import random
from turtle import *

class littleTurtle(Turtle):
    def setting(self, x, y):
        self.pen(shown=False, pendown=False)
        self.setpos(x, y)
        self.showturtle()
    
    def randColor(self):
        return random.randrange(0,255), random.randrange(0,255), random.randrange(0,255)
    
    def drawRectangle(self):
        randX = random.randrange(50,200)
        randY = random.randrange(50,200)

        randC = self.randColor()
        self.pencolor(randC)
        self.pensize(random.randint(1,10))
        self.pendown()

        for i in range(0,2):
            self.forward(randX)
            self.left(90)
            self.forward(randY)
            self.left(90)
        
        self.penup()
        
def createTurtle(x, y):
    lT1 = littleTurtle(shape="turtle", visible=False)
    lT1.setting(x, y)
    lT1.drawRectangle()

colormode(255)
onscreenclick(lambda x, y: createTurtle(x,y))
mainloop()
