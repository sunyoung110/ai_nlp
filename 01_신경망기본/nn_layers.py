
import numpy as np

## MatMul 노드

class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    # 순전파
    def forward(self,x):
        W, = self.params
        out = np.dot(x,W)
        self.x = x
        return out
    
    # 역전파
    def backward(self,dout):
        W, = self.params
        x = self.x
        dx = np.dot(dout,W.T)
        dW = np.dot(x.T,dout)
        self.grads[0][...] = dW   # 깊은 복사
        return dx


## 시그모이드 계층 : 역전파


class Sigmoid:
    def __init__(self):
        self.params,self.grads = [],[]
        self.out = None
        
    # 순전파    
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    # 역전파
    def backward(self,dout):
        dx = dout*self.out*(1 - self.out)  # y = sigmoid(x), y'= y*(1 - y) : sigmoid의 미분
        return dx


## Affine 계층 : 역전파

class Affine :  
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
    
    # 순전파
    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W) + b
        self.x = x
        return out
    
    # 역전파
    def backward(self,dout):
        W,b = self.params
        x = self.x
        dx = np.dot(dout,W.T)
        dW = np.dot(x.T,dout)
        db = np.sum(dout,axis = 0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx       


## Softmax with Loss 계층

class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [], []
        self.y = None    # softmax의 출력 값
        self.t = None    # 정답 레이블
        
    def softmax(self,x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))
        return x 
    
    # https://smile2x.tistory.com/entry/softmax-crossentropy-%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC 
    def cross_entropy_error(self,y, t):  
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 1e-7은 log(0)으로 무한대가 나오는걸 방지
          
    
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = self.cross_entropy_error(self.y, self.t)
        return loss
     
    def backward(self,dout=1):
        batch_size = self.t.shape[0]

        # dx = (self.y - self.t)/batch_size # 순수 Softmax계층 일경우
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


## 가중치 갱신

# 확률적 경사하강법(Stochastic Gradient Descent)
class SGD:
    def __init__(self,lr=0.01):
        self.lr = lr
    
    def update(self,params,grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]   


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
            




