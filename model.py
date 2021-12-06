class Model():
    def __init__(self):
        self.A = ""
        self.B = ""
        self.U = ""
        self.Y = ""
        self.G = ""
        self.L = ""
        self.T = ""
        self.L0 = []
        self.LG = []
        self.X0 = []
        self.XG = []
        self.Y0 = []
        self.YG = []
        self.TG = []
        self.VO = ""
        self.VG = ""

    def __str__(self):
        return f'A: {self.A},B :{self.B},U: {self.U},y^: {self.Y},G: {self.G},L: {self.L},T: {self.T},L0: {self.L0}, X0: {self.X0}, Y0: {self.Y0},LG: {self.LG}, XG: {self.XG},TG: {self.TG}, YG: {self.YG},VG: {self.VG},  V0: {self.VO}'
