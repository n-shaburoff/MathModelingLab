import scipy.integrate
import sympy as sp
import numpy as np
import sys
from model.main import Model

sys.setrecursionlimit(15000)


def calculateG():
    return sp.exp(-1 * (sp.Symbol('x') - sp.Symbol('z')) ** 2 / (4 * (sp.Symbol('t') - sp.Symbol('h')))) / sp.sqrt(
        4 * sp.pi * (sp.Symbol('t') - sp.Symbol('h')))


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
    C = float(99)
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
        self.v0 = model.V0
        self.vG = model.VG
        self.G = calculateG()

    ##Search Yinf
    def searchYinf(self, x, t):
        G_lamda = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G, 'numpy')
        u = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), self.u, 'numpy')

        G = lambda x, t, z, h: 0 if t - h <= 0 else G_lamda(x, t, z, h)
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)

        return scipy.integrate.dblquad(Gu, self.a, self.b, 0, self.T, epsabs=10 ** (-2), epsrel=10 ** (-2))[0]

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

    def makeAModulMatrix1(self):
        aModul = []
        for i in range(len(self.L0)):
            for j in range(len(self.X0)):
                G_lamda = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G, 'numpy')
                G = lambda x, t, z, h: 0 if t - h <= 0 else G_lamda(x, t, z, h)
                ##aModulElement = (mt.diffOperFunc(self.L0[i], self.G)).subs(sp.Symbol('t'), 0)
                aModulElement = lambda x, t: G(self.X0[j], 0, x, t)
                aModul.append(aModulElement)
        return np.transpose(np.array(aModul, ndmin=2))

    def makeAModulMatrix2(self):
        aModul = []
        for i in range(len(self.LG)):
            for j in range(len(self.TG)):
                G_lamda = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G, 'numpy')
                G = lambda x, t, z, h: 0 if t - h <= 0 else G_lamda(x, t, z, h)
                ##aModulElement = (mt.diffOperFunc(self.L0[i], self.G)).subs(sp.Symbol('t'), 0)
                aModulElement = lambda x, t: G(self.XG[j], self.TG[i], x, t)
                aModul.append(aModulElement)
        return np.transpose(np.array(aModul, ndmin=2))

    ## Make P

    def searchPmodul(self, A, i, j):
        A1_dot = self.dotMatrixFuncP(A[i - 1][0], np.transpose(A[j - 1][0]))
        A2_dot = self.dotMatrixFuncP(A[i - 1][1], np.transpose(A[j - 1][1]))
        ##A1_funct = lambda z, h: 0 if self.TG[j] - h <= 0 else A1_lambda(z, h)
        ##P1_funct=
        AC = self.a - self.C
        BC = self.b + self.C
        P1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.T, 0)
        P2 = self.searchIntegralMatrix(A2_dot, AC, BC, 0, self.T)

        Pmod = np.array(P1) + np.array(P2)
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
                elemMat = mat[i][j]
                interElemMat = scipy.integrate.nquad(elemMat, [[a, b], [t0, T]])[0]

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
        v0_func = np.array([[sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), v0, 'numpy')]])
        vG = sp.parse_expr(self.vG)
        vG_func = np.array([[sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), vG, 'numpy')]])
        A1_dot = self.dotMatrixFuncP(A[0][0], v0_func)
        A2_dot = self.dotMatrixFuncP(A[0][1], vG_func)
        AC = self.a - self.C
        BC = self.b + self.C
        A1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.T, 0)
        A2 = self.searchIntegralMatrix(A2_dot, AC, BC, 0, self.T)

        Av0 = np.array(A1) + np.array(A2)

        return Av0

    def searchAvG(self, A):
        v0 = sp.parse_expr(self.v0)
        v0_func = np.array([[sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), v0, 'numpy')]])
        vG = sp.parse_expr(self.vG)
        vG_func = np.array([[sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), vG, 'numpy')]])
        A1_dot = self.dotMatrixFuncP(A[1][0], v0_func)
        A2_dot = self.dotMatrixFuncP(A[1][1], vG_func)
        AC = self.a - self.C
        BC = self.b + self.C
        A1 = self.searchIntegralMatrix(A1_dot, self.a, self.b, -1 * self.T, 0)
        A2 = self.searchIntegralMatrix(A2_dot, AC, BC, 0, self.T)

        AvG = np.array(A1) + np.array(A2)

        return AvG

    def searchAv(self):
        A = self.makeAMatrix()
        Av = np.concatenate((self.searchAv0(A), self.searchAvG(A)), axis=0)
        return Av

    ## search Av A0

    def searchA0(self, x, t):
        A = self.makeAMatrix()

        A_1 = A[0][0]
        A1Shape = np.shape(A_1)
        for i in range(A1Shape[0]):
            for j in range(A1Shape[1]):
                A_1[i][j] = A_1[i][j](x, t)

        A_2 = A[1][0]
        A2Shape = np.shape(A_2)
        for i in range(A2Shape[0]):
            for j in range(A2Shape[1]):
                A_2[i][j] = A_2[i][j](x, t)

        A0 = np.concatenate((np.transpose(A_1), np.transpose(A_2)), axis=1)
        return A0

    def searchAG(self, x, t):

        A = self.makeAMatrix()
        A_1 = A[0][1]
        A1Shape = np.shape(A_1)
        for i in range(A1Shape[0]):
            for j in range(A1Shape[1]):
                A_1[i][j] = A_1[i][j](x, t)

        A_2 = A[1][1]
        A2Shape = np.shape(A_2)
        for i in range(A2Shape[0]):
            for j in range(A2Shape[1]):
                A_2[i][j] = A_2[i][j](x, t)

        AG = np.concatenate((np.transpose(A_1), np.transpose(A_2)), axis=1)
        return AG

    ## init P and Av

    def initPandAv(self):
        self.P = self.searchP()
        self.P = self.P.astype('float64')
        print(self.P)
        print('stop P')
        self.Av = self.searchAv()
        print('stop Av')
        self.Ys = self.makeYsVector()
        print('stop Ys')

    ## search u0 and uG
    def searchU0(self, x, t):
        A0 = self.searchA0(x, t)
        v0 = sp.parse_expr(self.v0)
        v0_func = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), v0, 'numpy')
        P = self.P
        U0 = np.dot(np.dot(A0, np.linalg.pinv(P)), (self.Ys - self.Av)) + v0_func(x, t)

        return U0

    def searchUG(self, x, t):
        AG = self.searchAG(x, t)
        vG = sp.parse_expr(self.vG)
        vG_func = sp.lambdify((sp.Symbol('z'), sp.Symbol('h')), vG, 'numpy')
        P = self.P
        UG = np.dot(np.dot(AG, np.linalg.pinv(P)), (self.Ys - self.Av)) + vG_func(x, t)
        return UG

    ## check det
    def chechInd(self):
        return False

    ## search Y0 YG and Y
    def searchY0(self, x, t):
        G_lamda = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G, 'numpy')
        u = lambda x, t: self.searchU0(x, t)[0][0]
        G = lambda x, t, z, h: 0 if t - h <= 0 else G_lamda(x, t, z, h)
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)

        return scipy.integrate.dblquad(Gu, self.a, self.b, -1 * self.T, 0, epsabs=10 ** (-2), epsrel=10 ** (-2))[0]

    def searchYG(self, x, t):
        G_lamda = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('z'), sp.Symbol('h')), self.G, 'numpy')
        u = lambda x, t: self.searchUG(x, t)[0][0]
        G = lambda x, t, z, h: 0 if t - h <= 0 else G_lamda(x, t, z, h)
        Gu = lambda z, h: G(x, t, z, h) * u(z, h)

        return scipy.integrate.dblquad(Gu, self.a - self.C, self.a, 0, self.T, epsabs=10 ** (-2), epsrel=10 ** (-2))[
                   0] + \
               scipy.integrate.dblquad(Gu, self.b, self.b + self.C, 0, self.T, epsabs=10 ** (-2), epsrel=10 ** (-2))[0]

    def searchY(self, x, t):
        if (self.P == None):
            self.initPandAv()
        y_inf = self.searchYinf(x, t)
        print(y_inf)
        yG = self.searchYG(x, t)
        print(yG)
        y0 = self.searchY0(x, t)
        print(y0)
        Y = y_inf + y0 + yG
        return Y