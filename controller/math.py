import scipy.integrate
import sympy as sp
import numpy as np
import sys
from MathModelingLab.model.main import Model

sys.setrecursionlimit(15000)

def calculateG():
    return sp.Heaviside((sp.Symbol('t') - sp.Symbol('h')) - sp.Abs(sp.Symbol('x') - sp.Symbol('z')))


class Math():
    L = None
    G = None
    u = None
    a = None
    b = None
    T = None
    TG = None
    X0 = None
    XG = None
    Y0 = None
    L0 = None
    LG = None
    YG = None
    C = float(10)
    Tm = -0.001
    P = None
    Av = None
    Ys = None

    def __init__(self, model: Model):
        self.L = model.L
        self.u = sp.parse_expr(model.U)
        self.a = model.A
        self.b = model.B
        self.T = model.T
        self.LG = model.LG
        self.L0 = model.L0
        self.XG = np.array(model.XG)
        self.X0 = np.array(model.X0)
        self.Y0 = np.array(model.Y0)
        self.YG = np.array(model.YG)
        self.TG = np.array(model.TG)
        self.v0 = model.VO
        self.vG = model.VG
        self.G = calculateG()

    ##Search Yinf
    def searchYinf(self, x, t):
        G = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G,
                        modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
        u = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), self.u, 'numpy')
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)

        return scipy.integrate.dblquad(Gu, 0, self.T, lambda h: self.a, lambda h: self.b, epsabs=10 ** (-3),
                                       epsrel=10 ** (-3))[0]

    ##MAKE Y
    def makeY0Vector(self):
        vectorY0 = []
        Y0shape = self.Y0.shape
        for i in range(Y0shape[0]):
            for j in range(Y0shape[1]):
                GuInter = self.searchYinf(self.X0[j], 0)
                Y0element = self.Y0[i][j] - GuInter
                vectorY0.append(float(Y0element))
        return np.transpose(np.array(vectorY0, ndmin=2))

    def makeYGVector(self):
        vectorYG = []
        YGshape = self.YG.shape
        for i in range(YGshape[0]):
            for j in range(YGshape[1]):
                GuInter = self.searchYinf(self.XG[j], self.TG[j])
                YGelement = self.YG[i][j] - GuInter
                vectorYG.append(float(YGelement))
        return np.transpose(np.array(vectorYG, ndmin=2))

    def makeYsVector(self):
        return np.concatenate((self.makeY0Vector(), self.makeYGVector()), axis=0)

    ##MAKE A
    def makeAMatrix(self):
        return np.array([[self.makeAModulMatrix1(), self.makeAModulMatrix1()],
                         [self.makeAModulMatrix2(), self.makeAModulMatrix2()]])

    def makeA(self):
        return np.vstack([np.hstack([self.makeAModulMatrix1(), self.makeAModulMatrix1()]),
                          np.hstack([self.makeAModulMatrix2(), self.makeAModulMatrix2()])])

    def makeAModulMatrix1(self):
        aModul = []
        for i in range(len(self.L0)):
            for j in range(len(self.X0)):
                ##aModulElement = (mt.diffOperFunc(self.L0[i], self.G)).subs(sp.Symbol('t'), 0)
                aModulElement = self.G.subs(sp.Symbol('t'), 0)
                aModulElement = aModulElement.subs(sp.Symbol('x'), self.X0[j])
                aModul.append(aModulElement)
        return np.transpose(np.array(aModul, ndmin=2))

    def makeAModulMatrix2(self):
        aModul = []
        for i in range(len(self.LG)):
            for j in range(len(self.TG)):
                ##aModulElement = (mt.diffOperFunc(self.LG[i], self.G)).subs(sp.Symbol('t'), self.TG[j])
                aModulElement = self.G.subs(sp.Symbol('t'), self.TG[j])
                aModulElement = aModulElement.subs(sp.Symbol('x'), self.XG[j])
                aModul.append(aModulElement)
        return np.transpose(np.array(aModul, ndmin=2))

    ## Make P

    def searchPmodul(self, A, i, j):
        A1_dot = np.dot(A[i - 1][0], np.transpose(A[j - 1][0]))
        A2_dot = np.dot(A[i - 1][1], np.transpose(A[j - 1][1]))
        A3_dot = np.dot(A[i - 1][1], np.transpose(A[j - 1][1]))
        ##A1_funct = lambda z, h: 0 if self.TG[j] - h <= 0 else A1_lambda(z, h)
        ##P1_funct=
        AC = self.a - self.C
        BC = self.b + self.C
        P1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.C, 0)
        P2 = self.searchIntegralMatrix(A2_dot, AC, self.a, 0, self.T)
        P3 = self.searchIntegralMatrix(A3_dot, self.b, BC, 0, self.T)
        Pmod = np.array(P1) + np.array(P2) + np.array(P3)
        return Pmod

    def searchP(self):
        A = self.makeAMatrix()
        P = np.concatenate((np.concatenate((self.searchPmodul(A, 1, 1), self.searchPmodul(A, 1, 2)), axis=1),
                            np.concatenate((self.searchPmodul(A, 2, 1), self.searchPmodul(A, 2, 2)), axis=1)), axis=0)
        return P

    ## integrall modul

    def searchIntegralMatrix(self, mat, a, b, t0, T):
        matShape = mat.shape
        for i in range(matShape[0]):
            for j in range(matShape[1]):
                elemMat_expr = mat[i][j].simplify()
                '''

                check=sp.integrate(elemMat_expr,(sp.Symbol('h'), t0, T)).simplify()
                check1 = sp.integrate(check, (sp.Symbol('z'), a, b))
                
                '''
                elemMat = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), elemMat_expr,
                                      modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
                interElemMat = \
                scipy.integrate.dblquad(elemMat, t0, T, lambda h: a, lambda h: b, epsabs=10 ** (-3), epsrel=10 ** (-3))[
                    0]
                mat[i][j] = float(interElemMat)

        return mat

    ## dot matrix for func

    def dotMatrixFuncP(self, mat1, mat2):
        mat1Shape = mat1.shape
        mat2Shape = mat2.shape
        row = mat1Shape[0]
        col = mat2Shape[1]
        mat = []
        for i in range(row):
            mat_col = []
            for j in range(col):
                matElement = lambda x, t: mat1[i][0](x, t) * mat2[0][j](x, t)

                mat_col.append(matElement)
            mat.append(mat_col)
        return np.array(mat)

    ## search Av
    def searchAv0(self, A):
        v0 = sp.parse_expr(self.v0)
        vG = sp.parse_expr(self.vG)
        A1_dot = np.dot(A[0][0], v0)
        A2_dot = np.dot(A[0][1], vG)
        A3_dot = np.dot(A[0][1], vG)
        AC = self.a - self.C
        BC = self.b + self.C
        A1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.C, 0)
        A2 = self.searchIntegralMatrix(A2_dot, AC, self.a, 0.0001, self.T)
        A3 = self.searchIntegralMatrix(A3_dot, self.b, BC, 0.0001, self.T)
        Av0 = np.array(A1) + np.array(A2) + np.array(A3)

        return Av0

    def searchAvG(self, A):
        v0 = sp.parse_expr(self.v0)
        vG = sp.parse_expr(self.vG)
        A1_dot = np.dot(A[1][0], v0)
        A2_dot = np.dot(A[1][1], vG)
        A3_dot = np.dot(A[1][1], vG)
        AC = self.a - self.C
        BC = self.b + self.C
        A1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.C, 0)
        A2 = self.searchIntegralMatrix(A2_dot, AC, self.a, 0.0001, self.T)
        A3 = self.searchIntegralMatrix(A3_dot, self.b, BC, 0.0001, self.T)

        AvG = np.array(A1) + np.array(A2) + np.array(A3)

        return AvG

    def searchAv(self):
        A = self.makeAMatrix()
        Av = np.concatenate((self.searchAv0(A), self.searchAvG(A)), axis=0)
        return Av

    ## search Av A0

    def searchA0(self):
        A = self.makeAMatrix()
        A0 = np.concatenate((np.transpose(A[0][0]), np.transpose(A[1][0])), axis=1)
        return A0

    def searchAG(self):
        A = self.makeAMatrix()
        AG = np.concatenate((np.transpose(A[0][1]), np.transpose(A[1][1])), axis=1)
        return AG

    ## init P and Av

    def initPandAv(self):
        self.P = self.searchP()
        self.P = self.P.astype('float64')
        print(self.P)
        print('stop P')
        self.Av = self.searchAv().astype('float64')
        print(self.Av)
        print('stop Av')
        self.Ys = self.makeYsVector().astype('float64')
        print(self.Ys)
        print('stop Ys')

    ## search u0 and uG
    def searchU0(self):
        A0 = self.searchA0()
        v0 = sp.parse_expr(self.v0)
        P = self.P
        U0 = np.dot(np.dot(A0, np.linalg.pinv(P)), (self.Ys - self.Av)) + v0

        return U0

    def searchUG(self):
        AG = self.searchAG()
        vG = sp.parse_expr(self.vG)
        P = self.P
        UG = np.dot(np.dot(AG, np.linalg.pinv(P)), (self.Ys - self.Av)) + vG
        return UG

    def searchU(self):
        A = self.makeA()
        P = self.P
        vG = sp.parse_expr(self.vG)
        v0 = sp.parse_expr(self.v0)
        v = np.array([[v0], [vG]])
        U = np.dot(np.dot(np.transpose(A), np.linalg.pinv(P)), (self.Ys - self.Av)) + v
        return U

    ## check det
    def chechInd(self):
        return False

    ## search Y0 YG and Y
    def searchY0(self, x, t):
        G = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G,
                        modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
        u0 = self.searchU0()[0][0].simplify()

        u = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), u0,
                        modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)

        return scipy.integrate.dblquad(Gu, self.Tm, 0, lambda h: self.a, lambda h: self.b, epsabs=10 ** (-3),
                                       epsrel=10 ** (-3))[0]

    def searchYG(self, x, t):
        G = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G,
                        modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
        uG = self.searchUG()[0][0].simplify()

        u = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), uG,
                        modules=['numpy', {'Heaviside': lambda arg: np.heaviside(arg, 1), 'Abs': np.abs}])
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)
        AC = self.a - self.C
        BC = self.b + self.C
        YG2 = \
        scipy.integrate.dblquad(Gu, 0, self.T, lambda h: AC, lambda h: self.a, epsabs=10 ** (-4), epsrel=10 ** (-4))[0]
        YG1 = \
        scipy.integrate.dblquad(Gu, 0, self.T, lambda h: self.b, lambda h: BC, epsabs=10 ** (-3), epsrel=10 ** (-3))[0]

        return YG1 + YG2

    def searchY(self, x, t):
        if (self.P.all() == None):
            self.initPandAv()
        y_inf = self.searchYinf(x, t)
        print(y_inf)
        yG = self.searchYG(x, t)
        print(yG)
        y0 = self.searchY0(x, t)
        print(y0)
        Y = y_inf + yG + y0
        return Y

    def searchAccuracy(self):
        Ys = self.makeYsVector()
        P = self.searchP().astype('float64')
        return np.dot(np.transpose(Ys), Ys) - np.dot(np.dot(np.transpose(Ys), P), np.dot(np.linalg.pinv(P), Ys))
