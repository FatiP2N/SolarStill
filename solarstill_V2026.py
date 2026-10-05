# -*- coding: utf-8 -*-
"""SolarStill - numerical model of solar stills (conventional, inclined, cascade, stepped).

Author: Fatima Belmehdi.
Runs locally (python solarstill.py) or in Jupyter / Google Colab.
Requirements: numpy, scipy, pandas, openpyxl, matplotlib, seaborn, iapws
(optional: tqdm for a progress bar, gspread + google-auth for online storing in Colab).
"""
import math
import datetime
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from iapws import IAPWS95   # equation of state for water
from iapws import SeaWater  # equation of state for seawater
from iapws import IAPWS97   # phase diagram of water
from iapws import humidAir as Hair  # equation of state for humid air
try:
    from tqdm import tqdm
except ImportError:          # tqdm is optional
    def tqdm(x):
        return x


@functools.lru_cache(maxsize=None)
def _read_sheet_cached(excel_path, sheet_name):
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def read_sheet(excel_path, sheet_name):
    """Read an Excel sheet once and reuse it (the file used to be re-read at every time step)."""
    return _read_sheet_cached(excel_path, sheet_name).copy()


class SolarStill:

    def __init__(self, excel_path,itr):
        self.kelv = 273.15
        self.conv = 3.6
        self.sig = 5.670374419 * 1e-8
        self.g = 9.81
        self.R = 8.31446261815324
        self.ma = 29 / 1e3
        self.mmw = 18.01528 / 1e3
        self.excel_path = excel_path
        self.ras = self.R / self.ma
        self.rw = self.R / self.mmw
        self.itr=itr
        self.listhcg = []
        self.ks = 1.0      # singular head loss at the inlet of the still (sudden widening)
        self.kv = 0.0      # head loss coefficient of the supply valve (0 = fully open); see kv_for_flow()
        self.feed = None   # measured feed flow in kg/h; if given, it replaces the hydraulic calculation
        self._hc_key = None
        # --- cost assumptions (can also be set in an optional Excel sheet named 'Cost', columns: Parameter, Value) ---
        self.interest = 0.02          # annual interest rate
        self.sunny_days = 340         # operating days per year (mean number of sunny days in Morocco)
        self.labor_step = 300         # labor cost per step, MAD (cascade and stepped stills)
        self.labor_length = 400       # labor cost per metre of basin, MAD (conventional and inclined stills)
        self.support_low = 200        # support cost, MAD (conventional and stepped stills)
        self.support_high = 600       # support cost, MAD (inclined and cascade stills)
        self.read_cost_assumptions()
        self.solver = "rk4"   # "rk4" (explicit Runge-Kutta 4) or "stiff" (implicit LSODA, needed for thin films / light basins)
        self.read_excel_data()
        #self.initialize_constants()


    def read_excel_data(self):
        geo = read_sheet(self.excel_path, 'design')
        self.geo  = geo
        self.df0  = geo.dim[0]
        self.df   = geo.dim[1]
        self.epb  = geo.dim[2]
        self.epi0 = geo.dim[3]
        self.epg  = geo.dim[4]
        self.l    = geo.dim[5]
        self.ns   = geo.dim[6]
        self.li0  = geo.dim[7]
        self.lh0  = geo.dim[8]
        self.b    = geo.dim[9]
        self.lh1  = geo.dim[10]
        self.lh2  = geo.dim[11]
        self.teta1 = np.pi * geo.dim[12] / 180
        self.teta2 = np.pi * geo.dim[13] / 180
        self.tetag = np.pi * (90 - geo.dim[12]) / 180
        self.teta  = np.pi * (90 - geo.dim[13]) / 180
        self.setup_geometry_parameters()
        self.read_material_properties()
        self.read_meteo_data()

    def setup_geometry_parameters(self):
        geo = self.geo
        if geo.dim[14] == 'With baffles':
            self.baff = 2 * geo.dim[20] * geo.dim[21]
            self.fs = (self.ns + geo.dim[21]) * 7
        else:
            self.fs = self.ns * 2
            geo.loc[20, 'dim'] = 0
            geo.loc[21, 'dim'] = 0
            self.baff = 0
        self.dc = geo.dim[19]
        self.d0 = geo.dim[16]
        self.epc = geo.dim[18]
        self.li = self.ns * self.li0
        self.ph = 2 * (self.ns * self.lh0 + self.lh1 + self.l)
        self.ah = (self.ns * self.lh0 + self.lh1) * self.l
        self.lb = self.ns * (self.li0 + self.lh0 + 2 * self.b) + self.lh1 + self.baff
        self.ab = self.lb * self.l
        self.lw = self.ns * (self.li0 + self.lh0) + self.lh1 + self.lh2
        self.lg = (self.ns * (self.lh0 + (np.cos(self.teta2)) * self.li0) + (self.lh1 + self.lh2)) / (
            np.cos(self.teta1))
        self.li = self.lw
        self.aw = self.lw * self.l
        self.ag = self.lg * self.l
        self.ai = self.li * self.l
        self.vsw = self.l * self.b * ((self.li0 * np.cos(self.teta2) * self.ns) + self.lh0 * self.ns + self.lh1) + (
                self.df0 * self.lh2) * self.l

    def read_material_properties(self):
        mat = read_sheet(self.excel_path, 'material')
        self.mater=mat.columns[6]+"-"+mat.columns[9]+"-"+mat.columns[12]
        self.s = mat.W[0]
        self.ew = mat.W[1]
        self.alw = mat.W[2] - self.s / 1e4
        self.tow = mat.W[3]
        self.f = mat.W[4]
        self.mtot = mat.W[5]
        self.mtotd = 0
        self.eg = mat.G[0]
        self.alg = mat.G[1]
        self.tog = mat.G[2]
        self.lamg = mat.G[3]
        self.rhoglass = mat.G[4]
        self.mg = self.ag * self.epg * self.rhoglass
        self.cpg = mat.G[5]
        self.eb = mat.B[0]
        self.alb = mat.B[1]
        self.lamb = mat.B[2]
        self.rhobasin = mat.B[3]
        self.mb = self.rhobasin * self.ab * self.epb
        self.cpb = mat.B[4]
        self.epsi = mat.B[6]
        self.lami0 = mat.Ins[0]
        self.lamc = mat.Ins[1]
        self.rhoins = mat.Ins[2]
        self.rhoinsc = mat.Ins[3]
        self.epi = self.epi0 + self.epc
        self.lami = self.epi / (self.epi0 / self.lami0 + self.epc / self.lamc)
        self.mi = self.ai * (self.rhoins * self.epi0 + self.rhoinsc * self.epc)
        self.ef = (1 / self.eg + 1 / self.ew - 1) ** -1
        self.ubi = (self.epb / self.lamb + self.epi / self.lami) ** -1

    def read_meteo_data(self):
        data = read_sheet(self.excel_path, 'Meteo')
        self.ta = data.Ta + self.kelv
        self.ig = data.Ig
        self.p = data.P[0]
        self.v = np.mean(data.V) / self.conv
        self.w = data.H[0]
        self.tf = np.zeros([len(data.t)])
        for i in range(len(data.t)):
            self.tf[i] = self.t_flt(data.t[i])
        self.t = np.linspace(min(self.tf), max(self.tf), len(self.tf))
        self.y = np.array(self.ta).reshape(len(self.ta), )
        self.tamb = interp1d(self.t, self.y)
        self.y1 = np.array(self.ig).reshape(len(self.ig), )
        self.ig1 = interp1d(self.t, self.y1)
        self.gps=data.iloc[6,9]
        self.month=data.iloc[7,9]



    def t_flt(self, dt):# Time to float : transform datatime to float
        a = (dt.hour)
        b = (dt.minute)
        fdt = a + b / 60
        return fdt

    def time1(self, t0):# float to Time  : transform float to datatime
        hour = int(t0)
        mns = int((t0 - hour) * 60)
        t0 = datetime.time(hour, mns)
        return t0.strftime("%H:%M")

    def simulate(self):
        itr = self.itr
        t1 = np.linspace(min(self.tf), max(self.tf), len(self.tf) * itr)
        return t1

    def technical_sheet(self):
        geo = self.geo
        mat = read_sheet(self.excel_path, 'material')
        self.device = geo.dim[15]
        if self.device == "Conventional":
            self.tetai = 0
            self.li == 0
            self.supp = self.support_low
            self.labor = self.lb * self.labor_length
            self.df = (self.df + self.lg * np.sin(self.tetag)) / 2
        elif self.device == "Stepped":
            self.supp = self.support_low
            self.labor = self.ns * self.labor_step
        elif self.device == "Inclined":
            self.tetai = self.tetag
            self.lh0 = 0
            self.supp = self.support_high
            self.labor = self.lb * self.labor_length
        elif self.device == "Cascade":
            self.supp = self.support_high
            self.labor = self.ns * self.labor_step


        print("\033[1m\033[4mTECHNICAL SHEET\033[0m\033[0m", "\n", "\033[1mSolar still type:\033[0m", self.device,
              "Solar still", self.geo.dim[14])
        print("\033[32mMaterial:\033[0m", "\n", 'Cover in', mat.columns[6], "\n", 'Absorber in', mat.columns[9], "\n",
              'and Insulation in', mat.columns[12])
        technical_sheet = geo[['Description', 'dim', 'Unit']]
        technical_sheet = technical_sheet.rename(columns={'dim': 'Value'})
        print(technical_sheet)
        # Solar Still Price
        Price = round((self.mg * mat.G[6] + self.mb * mat.B[5] + self.ai * (
                mat.Ins[4] * self.rhoins * self.epi0 + self.rhoinsc * self.epc * mat.Ins[5]) + self.labor + self.supp), 2)
        USprice = round(0.1 * (self.mg * mat.G[6] + self.mb * mat.B[5] + self.ai * (
                mat.Ins[4] * self.rhoins * self.epi0 + self.rhoinsc * self.epc * mat.Ins[5]) + self.labor + self.supp), 2)
        print("\033[34mPrice(initial investment) = \033[0m", "\n", Price, 'MAD', USprice, '$US')
        self.usprice=USprice

    def plot_device(self):
        self.technical_sheet()
        self.device = self.geo.dim[15]
        xb = np.zeros(2 * self.ns + 5)
        yb = np.zeros(2 * self.ns + 5)
        xv = np.zeros(2)
        yv = np.zeros(2)
        xi = np.zeros(6)
        yi = np.zeros(6)
        xb[0:5] = [self.epi, self.epi, self.lh2 + self.epi, self.lh2 + self.epi, self.epi + self.lh1 + 2 * self.lh2]
        yb[0:5] = [self.df0 + self.epi, self.epi, self.epi, self.df0 + self.epi, self.df0 + self.epi]

        for i in range(4, 2 * self.ns + 4, 2):
            xb[i + 1] = xb[i] + self.li0 * np.cos(self.teta2)
            xb[i + 2] = xb[i + 1] + self.lh0
            yb[i + 1] = yb[i] + self.li0 * np.sin(self.teta2)
            yb[i + 2] = yb[i + 1]

        if self.geo.dim[13] == 0:
            self.sss = self.lb * np.sin(self.teta1)
            self.lg = self.sss
        elif self.geo.dim[13] >= 90:
            self.sss = self.df
        else:
            self.sss = self.epi + self.df

        xv = [0, xb[2 * self.ns + 4]]
        yv = [xv[0] + self.df + self.df0 + self.epi, self.sss + (
                yb[2 * self.ns + 4] + (xb[2 * self.ns + 4] - xb[2 * self.ns + 3]) / (
                (xb[2 * self.ns + 3] - xv[0]) / (yb[2 * self.ns + 4] + self.df - yv[0])))]

        xi = [0, 0, xi[1] + self.lh2 + self.epi * 2, xi[1] + self.lh2 + self.epi * 2, xb[4], xb[2 * self.ns + 4]]
        if self.device == "Stepped" or self.device == "Conventional":
            xi = [0, 0, xi[1] + self.lh2 + self.epi * 2, xb[2 * self.ns + 4]]
            yi = [yb[0], 0, 0, 0]
        else:
            yi = [yb[0], 0, 0, yi[2] + self.df0 - self.epi, yi[2] + self.df0 - self.epi,
                  yb[2 * self.ns + 4] - 2 * self.epi]

        plt.plot(xb, yb, label='Collector', c='r')
        plt.plot(xb[3:len(xb)], yb[3:len(xb)], label='Absorber', marker='.', c='blue')
        plt.plot(xv, yv, label='Cover')
        plt.plot(xi, yi, label='Insulation', c='black', linestyle=':')
        if self.device == "Stepped":
            plt.plot(xb[3:len(xb)] + self.lh0 / 4, yb[3:len(xb)] - self.li0 / 2, c='black', linestyle=':')
        if self.geo.dim[14] == 'With baffles':
            chicx = np.linspace(xb[3], xb[len(xb) - 1], len(xb) * 4)
            ybch = np.array(yb[3:len(xb)].reshape(len(yb) - 3, ))
            chic = interp1d(xb[3:len(xb)], ybch)
            plt.scatter(chicx, chic(chicx) + 0.01, c='brown', marker='2')


        if self.geo.dim[14]== 'With baffles':
            plt.title(self.device+ ' Solar Still ' + self.geo.dim[14],fontsize=12)
        else :
            plt.title(self.device+ ' Solar Still ' ,fontsize=12)

        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.show()

#####################################################################################################################################

##############################################################################
    # Latent Heat of vaporisation of saltwater kJ/kg
    def Lv(self, Tw0):
        T = Tw0 - self.kelv
        if T >= 70:
            Lv = 3.1615 * 1e3 * (1 - (7.616 * 1e-4 * T))
        else:
            Lv = 2.4935 * 1e3 * (
                    1
                    - (9.4779 * 1e-4 * T)
                    + 1.3132 * 1e-7 * T ** 2
                    + 4.7974 * 1e-9 * T ** 3
            )
        return Lv

    # Define etoile for Hollands correlation
    def etoile(self, x):
        return max(x, 0)  # return 0 or positive


    def secondegree(self, a, b, c):
        # Solve the quadratic equation ax**2 + bx + c = 0 for the film water velocity
        d = (b ** 2) - (4 * a * c)
        if d < 0:
            print("delta < 0", a, b, c)
        else:
            sol1 = (-b - math.sqrt(d)) / (2 * a)
            sol2 = (-b + math.sqrt(d)) / (2 * a)
            return max(sol1, sol2)  # return only positive solution

    def read_cost_assumptions(self):
        """Optional sheet 'Cost' with two columns (Parameter, Value): interest, sunny_days, labor_step, labor_length, support_low, support_high."""
        try:
            cost = read_sheet(self.excel_path, 'Cost')
        except Exception:
            return
        for name, value in zip(cost.iloc[:, 0], cost.iloc[:, 1]):
            if str(name).strip() in ('interest', 'sunny_days', 'labor_step', 'labor_length', 'support_low', 'support_high'):
                setattr(self, str(name).strip(), float(value))

    def crf_(self, i, n):
        return i * (1 + i) ** n / ((1 + i) ** n - 1)

    def sff_(self, i, n):
        return i / ((1 + i) ** n - 1)

    def annual_cost_E(self, Ps, Mt, Sv, i, n):
        crf = self.crf_(i, n)
        sff = self.sff_(i, n)
        UA = (Ps + Ps * Mt) * crf - Sv * sff
        return UA

    def annual_cost_total(self, components, i):
        AC = 0
        for E in components:
            Ps = E["Ps"]
            Mt = E["Mt"]
            Sv = E["Sv"]
            n = E["n"]
            AC += self.annual_cost_E(Ps, Mt, Sv, i, n)
        return AC

    def cpk(self, me, i=None, lifetime=None, Svg=1e-3,Svb=1e-4,Svi=1e-3):
        """
        Cost per kg of distilled water (USD/kg).
        i        : annual interest rate (default: self.interest)
        lifetime : if given, the same lifetime (years) is used for every component;
                   otherwise each component keeps the lifetime of the 'material' sheet.
        """
        if i is None:
            i = self.interest
        mat1=read_sheet(self.excel_path, 'material')
        components = [
            {
                "Ps": 0.1 * mat1.G[6] * self.mg,
                "Mt": 0.03,
                "Sv": Svg * self.usprice,
                "n": mat1.G[7],
            },  # cover
            {
                "Ps": 0.1 * mat1.B[5] * self.mb,
                "Mt": 0.02,
                "Sv": Svb * self.usprice,
                "n": mat1.B[7],
            },  # absorber
            {
                "Ps": 0.1 * self.ai * (mat1.Ins[4] * self.rhoins * self.epi0 + self.rhoinsc * self.epc * mat1.Ins[5]),
                "Mt": 0.015,
                "Sv": Svi * self.usprice,
                "n": mat1.Ins[7],
            },  # insulation
            {
                "Ps": 0.1 * (self.labor + self.supp),
                "Mt": 0.001,
                "Sv": 0,
                "n": 50,
            },  # labor and support (concrete): amortised, no salvage value
        ]
        if lifetime is not None:
            for E in components:
                E["n"] = lifetime
        AC_total = self.annual_cost_total(components, i)
        self.annual_cost = AC_total                # USD per year
        daily_cost = AC_total / self.sunny_days
        CPK = daily_cost / me      # USD per kg (the former extra 0.1 factor was a double MAD->USD conversion)
        return CPK

    def cpk_standard(self, me):
        """Cost per kg with the assumptions most used in the literature: 12 % interest, 10-year lifetime."""
        return self.cpk(me, i=0.12, lifetime=10)


    def hcgw(self, Tw0, Tg0):

        #Free convective Heat of the humid air circulating inside the solar still between
        #the condensation surface and the evaporation surface. Calculated using different correlation:
        #R.V Dunkle, Holland et al, Chen et al, Zheng Hongfei et al

        if not isinstance(Tw0, float):
            raise NotImplementedError("Incoming out of bound", Tw0)
        if self._hc_key == (Tw0, Tg0):
            return self._hc_val

        Psw = (
                100
                * np.exp(
            31.3756 - (6745.09 / Tw0) - 4.8489 * np.log((Tw0) / 100) - 0.000544 * self.s
        )
        )

        Pg = self.PSat(Tg0)
        #print(Tg0-self.kelv)

        deltaTwg = ((Tw0 - Tg0) + ((Psw - Pg) * (Tg0) / (268.9 * 1e3 - Psw)))
        Corr_choice=read_sheet(self.excel_path, 'Corr')
        Corr=Corr_choice.Therm_mod[0]

        if Corr == "R.V Dunkle":
            hcgw = 0.884 * (self.etoile(deltaTwg)) ** (1 / 3)
        else:
            Th = (Tg0 + Tw0) / 2

            w = IAPWS95(T=Th, P=self.p)
            air = Hair.Air(T=Th, P=self.p)  # dried air
            Ha = Hair.HumidAir(T=Th, P=self.p, W=self.w)  # humid air

            # Properties of humid air
            beta1 = Ha.alfav  # coefficient of thermal expansion
            rhom = Ha.rho  # density of humid air
            vmaw = (Ha.A * air.mu + Ha.W * w.mu) / (rhom)  # dynamic viscosity of humid air mu/rho
            lama = Ha.A * air.k + Ha.W * w.k  # conductivity of humid air
            alfam = lama / (rhom * 1e3 * Ha.cp)  # thermal diffusivity of humid air

            if Corr == "Hollands et al":
                Ra = self.g * (beta1 * (Tw0 - Tg0) * self.df ** 3) / (vmaw * alfam)  # Rayleigh
                Nuwg = 1 + 1.44 * self.etoile(1 - (1708 / Ra * np.cos(self.teta1)) * (1- (1708 * ((np.sin(1.8 * self.teta1))) ** (1 / 6))/ Ra * np.cos(self.teta1) + self.etoile(((Ra * (np.cos(self.teta1) / 5830)) ** (1 / 3)) - 1)))
                # Nusselt pour surface plane inclinée, Holland
                hcgw = Nuwg * lama / self.df

            elif Corr == "Chen et al":
                Ra0 = self.g * (np.cos(self.teta1) * beta1 * (Tw0 - Tg0) * self.df ** 3) / (vmaw * alfam)
                hcgw = (0.2 * (Ra0) ** 0.26) * lama / self.df

            elif Corr == "Zheng Hongfei et al":
                M = (self.ma * self.p * 1e6) / (self.ma - self.mmw)
                deltaTwg1 = (Tw0 - Tg0) + ((Psw - Pg) * Tg0 / (M - Psw))
                Ra1 = self.g * (np.cos(self.teta1) * beta1 * deltaTwg1 * self.df ** 3) / (vmaw * alfam)
                hcgw = (0.2 * Ra1 ** 0.26) * lama / self.df

        self.listhcg.append(hcgw)
        self._hc_key, self._hc_val = (Tw0, Tg0), hcgw
        return hcgw

    def PSat(self,T):

        #Saturation vapor pressure in MPa (multiply by 1e6)
        #https://doi.org/10.1115/1.483186 Eq:(2.13)
        #Iapws Formula

        if T <= 273.15 or T > 647.096:
            raise NotImplementedError("Incoming out of bound", T)

        n = (0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
             0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
             -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
             0.65017534844798E+03)
        ta = T + n[9] / (T - n[10])
        A = ta ** 2 + n[1] * ta + n[2]
        B = n[3] * ta ** 2 + n[4] * ta + n[5]
        C = n[6] * ta ** 2 + n[7] * ta + n[8]
        return 1e6*(2 * C / (-B + (B ** 2 - 4 * A * C) ** 0.5)) ** 4

        #Dupré's formula simplified by Rankine
        #P=np.exp(25.317-(5144/(T))
        #T in °C & P in Pascal
        #https://books.google.co.ma/books?id=oxdEEAAAQBAJ&pg=PA482&redir_esc=y#v=onepage&q&f=false


    def TSat_P(self,P):
        #
        # Saturation temperature in K
        #https://doi.org/10.1115/1.483186 Eq:(2.14)
        #
        # Check input parameters
        if P < 611.212677 / 1e6 or P > 22.064:#MPa
            raise NotImplementedError("Incoming out of bound")

        n = (0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
             0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
             -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
             0.65017534844798E+03)
        be = P ** 0.25
        E = be ** 2 + n[3] * be + n[6]
        F = n[1] * be ** 2 + n[4] * be + n[7]
        G = n[2] * be ** 2 + n[5] * be + n[8]
        D = 2 * G / (-F - (F ** 2 - 4 * E * G) ** 0.5)
        return (n[10] + D - ((n[10] + D) ** 2 - 4 * (n[9] + n[10] * D)) ** 0.5) / 2

    def Tr(self,T,P,x):#dew point

        #Dew temperature calculated from Temperature and relative humidity x
        #using Heinrich Gustav Magnus-Tetens' formula

        # Check input parameters
        if T >= 60  or P <= 0 or x<0.01 or x>1:
            raise NotImplementedError("Incoming out of bound",T,x)
        a=17.27
        b=237.7
        f=np.log(x)+(a*T/(T+b))
        return b*f/(a-f)

# computes the function x -> x/2 + log(a+b.x) which zero corresponds to 1/sqrt(f_D)
    def f_(self,eps,D_h,Re,x):
        #return f and its derivative f' as a tuple
        a = eps/3.7*D_h
        b= 2.51/Re
        return (x/2 + np.log(a+b*x),1/2 + b/(a+b*x))
    #Newton's methode to approximate the friction factor fD
    def newton(self,f_,eps,D_h,Re,x):
        fx, dfx = self.f_(eps,D_h,Re,x)
        while np.abs(fx)>1e-8:
            x= x - fx/dfx
            fx,dfx = f_(eps,D_h,Re,x)
        return 1/x**2

    def hcond (self,Tg0,Tw0,Ta0):

        #Enthalpy of condensation according to Dhir, V. K., and J. H. Lienhard, J. Heat Transfer, 93, 97,1971.

        if Tg0<=(self.Tr(Ta0 - self.kelv,self.p,self.w)+self.kelv):

            Tsat=self.TSat_P(self.p)
            liq=IAPWS97(T=((Tsat+Tg0)/2 ),P=self.p,x=0)
            vap=IAPWS97(T=Tsat,P=self.p,x=1)
            rhol=liq.rho
            rhov=vap.rho
            laml=liq.k
            mul=liq.mu
            lat=self.Lv(Tsat)+0.68*liq.cp*(Tsat-(Tg0))#Rohsenow, W. M., Trans. ASME, 74, 969, 1952
            hcond=(((rhol-rhov)*rhol*self.g*(np.cos(self.teta1))*laml**3 * lat)/((Tsat-(Tg0))*mul*self.lg))
            return 0.943*(1e-4*hcond)**(1/4)
        else:
            return 0

    def HrtoHs(self,T):#empiric foormula of Nadeau et Puiggali transforming specific humidity to relative!!!
        Hs=622*self.w*self.PSat(T) / (101325- self.PSat(T)*(self.w/1e3))
        return Hs

    def musw(self,T,P,S):
        w=IAPWS95(P=self.p, T=T)
        T=T-self.kelv
        A=1.541+1.998*1e-2*T-9.52*1e-5*T**2
        B=7.974-7.561*1e-2*T+4.724*1e-4*T**2
        S=(35.16504*S/35)/1000
        mu=w.mu*(1+ A*S + B* S**2)
        return mu

    def Nubwi(self,Tb0,Tw0):
        if not isinstance(Tw0, float):
            raise NotImplementedError("Incoming out of bound", Tw0,Tb0)
        #  Nusselt number for inclined surface hot surface upwards basin-salt water
        #  Churchill and Chu
        #  Churchill, S. W., and H. H. S. Chu, Int. J. Heat Mass Transfer, 18, 1323, 1975.

        sw=SeaWater(P=self.p, T=Tw0, S=self.s*1e-3 )
        mucor=self.musw(Tw0,self.p,self.s)
        Pr=mucor*sw.cp/sw.k
        #print("i",Pr)
        #alfasw=sw.k/(sw.rho*sw.cp)
        nusw=mucor/(sw.rho)
        Rai=(self.g*np.cos(self.teta) * sw.alfav * (self.ns*self.li0)**3 * self.etoile(Tb0-Tw0)/((nusw)**2)) * Pr
        #Churchill and Chu : A correlation that may be applied over the entire range of Rai and 0<=teta2<=60°
        Rai1=(1+(0.387*Rai**(1/6))/(1+(0.492/(Pr))**(9/16))**(8/27))**2
        #print("Rai=",Rai, Rai1)
        return Rai1

    def Nubwh(self,Tb0,Tw0):
        #Nusselt number for horizontal surface hot surface upwards steam between the glass and the salt water surface
        #Lloyd, J. R., and W. R. Moran, J. Heat Transfer, 96, 443,1974.

        sw=SeaWater(P=self.p, T=Tw0, S=self.s*1e-3 )
        mucor=self.musw(Tw0,self.p,self.s)
        #print(mucor)
        Pr=mucor*sw.cp/sw.k
        #print("h",Pr, sw.k,Tw0-self.kelv)
        nusw=mucor/(sw.rho)
        Rah=(self.g * sw.alfav * (self.ah/self.ph)**3 * self.etoile(Tb0-Tw0)/((nusw)**2)) * Pr
        if (Rah<1e4):
            c,m= 1,0
        elif  Rah>=1e4 and Rah<=1e7:
            c,m=0.54,(1/4)
        elif Rah>1e7:
            c,m=0.15,(1/3)
        Rah1= c*Rah**m
        #print("Rah=",Rah, Rah1,sw.alfav )
        return Rah1


    def hcga(self):

        #Enthalpy of convection due to air speed
        #A. El-Sebaii " Effect of Wind Speed on Active and Passive Solar Stills",
        #Energy Consversion and Management, 45, 1187-1204, 2004.

        if self.v<= 5:
            h=3.*self.v+2.8
        else:
            h=6.15*self.v**0.8
        return h

    def hvap_(self,Tg0,Tw0):

        #Enthalpy of vaporisation of water

        #Pw=np.exp(25.317-(5144/(Tw0)))#Vapor pressure empirical correlation
        #Pg=np.exp(25.317-(5144/(Tg0)))#Vapor pressure empirical correlation
        Pw=self.PSat(Tw0)#Pressue of saturation at the level of water
        Pg=self.PSat(Tg0)#Pressue of saturation at the level of the cover
        hvap_= self.hcgw(Tw0,Tg0) * (16.276*1e-3)*(Pw-Pg)/(Tw0-Tg0)
        return hvap_


    def Tini(self,Tb0,Ti0):
        """
        Temperature of the internal face of insulation (Array Tin)
        """
        return (Tb0+(self.ai*self.lami*self.epb)/(self.ab*self.lamb*self.epi)*Ti0)/(1+((self.ai*self.lami*self.epb)/(self.ab*self.lamb*self.epi)))

    def Tinx(self,Ta0,Tb0):
        """
        Temperature of the external face of insulation (Array Tiex)
        """
        if self.device == "Conventional":
            h=0
        else:
            h=self.hcga()
        return (Tb0*self.ai*self.ab*(self.lami/self.epi)*(self.lamb/self.epb)- Ta0*self.ag*h*(self.ab*(self.lamb/self.epb)+self.ai*(self.lami/self.epi)))/(self.ai*self.ab*self.lami*self.lamb/(self.epi*self.epb) - self.ag*h*(self.ab*self.lamb/self.epb - self.ai*self.lami/self.epi))

    def feed_flow(self, Tw0):
        """
        Mass flow of salt water fed to the still, in kg/h.
        The tank empties through the supply valve (Torricelli) and the flow is reduced by the head losses:
        kv (valve), ks (inlet of the still) and fs (steps and baffles inside the still).
        The same flow crosses the whole still (mass conservation).
        """
        if self.feed is not None:
            return float(self.feed)
        sw=SeaWater(P=self.p, T=Tw0+0, S=self.s*1e-3)
        Mw=sw.rho *(self.vsw)
        Hw=4*max(self.mtot-Mw-self.mtotd,0)/ (sw.rho*np.pi*self.dc**2)   # water height in the tank (m)
        v_valve=(2*self.g*Hw/(1+self.kv+self.ks+self.fs-(self.d0**4/self.dc**4)))**(1/2)   # m/s
        return 3600*sw.rho*(np.pi*self.d0**2/4)*v_valve

    def kv_for_flow(self, target, Tw0=300.0):
        """Valve head loss coefficient kv that gives the feed flow 'target' (kg/h)."""
        sw=SeaWater(P=self.p, T=Tw0, S=self.s*1e-3)
        Mw=sw.rho *(self.vsw)
        Hw=4*max(self.mtot-Mw-self.mtotd,0)/ (sw.rho*np.pi*self.dc**2)
        v_valve=target/(3600*sw.rho*(np.pi*self.d0**2/4))
        return 2*self.g*Hw/v_valve**2-1-self.ks-self.fs+(self.d0**4/self.dc**4)

    def qf_(self,Ta0,Tw0,mtot):
        """
        Heat carried by the film of water circulating through the solar still, in kJ/(h.K):
        qf = feed flow (kg/h) x Cp (kJ/kg.K). The water enters at ambient temperature and leaves at Tw.
        """
        sw=SeaWater(P=self.p, T=Tw0+0, S=self.s*1e-3)
        self.mdot=self.feed_flow(Tw0)
        return self.mdot*sw.cp


    def Bilan(self,t,Tan,Ta0,Ig0):
        Tg0=Tan[0]
        Tw0=Tan[1]
        Tb0=Tan[2]
        me0=Tan[3]
        #salt water in the solar still
        sw=SeaWater(P=self.p, T=Tw0+0, S=self.s*1e-3)
        Cpw=sw.cp # Cp water in kj/kg.K
        Mw=sw.rho *(self.vsw)# Water mass in the solar still
        hrgsky=self.sig*self.eg #radiation sky-cover
        hrgw=self.sig*self.ef #radiatin water-cover
        Tsky=0.0552*Ta0**1.5 #Sky temperature (Swinbank), Ta0 in kelvin

        if self.li0==0:
            hcbw=sw.k*self.Nubwh(Tb0,Tw0)/(self.ab/(2*(self.lb+self.l)))
            qf=0
            Mw=self.mtot-self.mtotd

        elif self.lh0==0:
            hcbw=sw.k*self.Nubwi(Tb0,Tw0)/self.lb
            qf=self.qf_(Ta0,Tw0,self.mtot)
        else:
            #hcbw= sw.k*(abs(Nubwh(Tb0,Tw0)**5 + Nubwi(Tb0,Tw0)**5)**(1/5))/Lb
            hcbw=sw.k*(self.Nubwh(Tb0,Tw0)/(self.ab/self.ph) + self.Nubwi(Tb0,Tw0)/self.lb)
            qf=self.qf_(Ta0,Tw0,self.mtot)
            #print("hcbw=",hcbw)

        #Cover balance
        cg=self.conv*((self.hcga()-self.hcond(Tg0,Tw0,Ta0))*self.ag+(self.hcgw(Tw0,Tg0)+self.hvap_(Tg0,Tw0))*self.aw)/(self.mg*self.cpg)
        bg=self.conv*(self.ag*(self.hcga()*Ta0 -self.hcond(Tg0,Tw0,Ta0)*Tw0+ Ig0*self.alg)+self.aw*Tw0*(self.hcgw(Tw0,Tg0)+self.hvap_(Tg0,Tw0))+ self.ag*hrgsky*Tsky**4 + self.aw*hrgw*Tw0**4)/(self.mg*self.cpg)
        dg=self.conv*(self.ag*hrgsky + self.aw*hrgw)/(self.mg*self.cpg)

        #Water balance
        cw=(self.conv*((self.hcgw(Tw0,Tg0)+self.hvap_(Tg0,Tw0))*self.aw+(hcbw*self.ab))+qf)/(Mw*Cpw)
        #print(qf)
        bw=(self.conv*(self.aw*(Tg0)*(self.hcgw(Tw0,Tg0)+self.hvap_(Tg0,Tw0))+ hcbw*self.ab*Tb0 + Ig0*self.alw*self.tog*self.aw + hrgw*self.aw*((Tg0)**4))+ qf*(Ta0))/(Mw*Cpw)
        dw=self.conv*hrgw*self.aw/(Mw*Cpw)

        #Absorber balance
        Ti0=self.Tinx(Ta0,Tb0)
        Ti0=(Ti0+self.Tini(Tb0,Ti0))/2
        Ubi=1/((self.epi/self.lami)+(self.epb/self.lamb))
        cb=self.conv*(self.ab*(hcbw+Ubi))/(self.mb*self.cpb)
        bb=self.conv*(self.ab*((hcbw*Tw0)+(Ti0*Ubi))+Ig0*self.alb*self.ab*self.tog*self.tow)/(self.mb*self.cpb)
        db=0

        #Evaporated water
        cm=0
        bm=self.conv*(self.ab*self.hvap_(Tg0,Tw0)*self.etoile(Tw0-Tg0))/(self.Lv(Tw0))
        dm=0
        c= np.array((cg,cw,cb,cm),float)
        b=np.array((bg,bw,bb,bm),float)
        d=np.array((dg,dw,db,dm),float)
        #    dico={'B':b,'C':c, 'D':d}
        #    print (dico)
        return   (- d*((Tan)**4) - c*(Tan) +b) #f(t,Temp(t))=dT/dt= -d(t,T(t)) *T(t)**4 - c(t,T(t))*T(t)+b(t,T(t))


 #########################################################################
    def ss_performances(self, itr=None):
        if itr is not None:
            self.itr=itr
        itr=self.itr
        t1=self.simulate()
        self.Tin=np.zeros([len(t1)])
        self.Tiex=np.zeros([len(t1)])
        #temperature in K at t=0
        self.Temp=np.zeros([4,len(t1)])
        Ta0=self.ta[0]
        Ig0=self.ig[0]
        self.Temp[0,0]=Ta0+0.5
        Tg0=self.Temp[0,0]
        self.Temp[1,0]=Tg0+1
        Tw0=self.Temp[1,0]
        self.Temp[2,0]=Tw0+0.01
        Tb0=self.Temp[2,0]
        self.Temp[3,0]=0# md ditilated water at t=0
        me=self.Temp[3,0] #evaporated water
        Ti0=Ta0-0.5
        self.Tin[0]=Tb0-1
        self.Tiex[0]=Ti0

        #Solar still efficiency
        efficiency0=0
        efficiency1=(((self.ig[0]))*self.alb*self.ab*self.tog*self.tow*self.conv)
        incident=(((self.ig[0]))*self.ab*self.conv)

        #Runge Kutta RK4
        for i in tqdm(range(len(t1)-1)):
            h =(t1[i+1]-t1[i])# in hours
            if self.solver == "stiff":
                sol = solve_ivp(lambda tt, y: self.Bilan(tt, y, Ta0, Ig0), (0, h), self.Temp[:,i], method="LSODA", rtol=1e-4, atol=1e-5)
                self.Temp[:,i+1] = sol.y[:,-1]
            else:
                k1= self.Bilan(h,self.Temp[:,i],Ta0,Ig0)
                k2= self.Bilan((h + h / 2.),(self.Temp[:,i] + (k1 * h) / 2.),Ta0,Ig0)
                k3= self.Bilan((h + h / 2.),(self.Temp[:,i] + (k2 * h) / 2.),Ta0,Ig0)
                k4= self.Bilan((h + h ),(self.Temp[:,i]+ (k3 * h)),Ta0,Ig0)
                self.Temp[:,i+1] = self.Temp[:,i]  + ((h / 6.) * (k1 + 2*k2 + 2*k3 + k4))
            #Temp[3,i]=(conv*h*hvap_(Temp[0,i],Temp[1,i])*Aw)/Lv(Temp[1,i]) #
            self.Tin[i+1]=self.Tini(self.Temp[2,i],self.Tinx(Ta0,self.Temp[2,i]))
            self.Tiex[i+1]=self.Tinx(Ta0,self.Temp[2,i])
            Ti0=self.Tin[i+1]
            Ta0=self.tamb(t1[i+1])
            Ig0=self.ig1(t1[i+1])
            sw=SeaWater(P=self.p, T=self.Temp[1,i], S=self.s*1e-3)
            if self.li0==0:   # conventional still: no circulating film, only evaporated water leaves the basin
                self.mtotd+=(self.Temp[3,i+1]-self.Temp[3,i])
            else:
                self.mtotd+=(self.Temp[3,i+1]-self.Temp[3,i] + h*self.feed_flow(self.Temp[1,i]))   # distilled water + water fed to the still
            #W=W+0.001/itr #must be improved
            #print(me, me/(itr*h))
            efficiency0+=((self.Temp[3,i+1]-self.Temp[3,i])*self.Lv(self.Temp[1,i+1]))
            efficiency1+=((Ig0)*self.alb*self.ab*self.tog*self.tow*self.conv)/self.itr
            incident+=((Ig0)*self.ab*self.conv)/self.itr
        # efficiency relative to the energy absorbed by the basin (original definition)
        self.efficiency= 100*efficiency0/(efficiency1)
        # efficiency relative to the incident solar energy (definition used for experimental data and in the literature)
        self.efficiency_incident= 100*efficiency0/(incident)
        #print("Solar still performance calculated")

    def plot_performances(self):
        t1=self.simulate()
        #print(Temp[3,len(t1)-1])
        #distilled water
        self.me=self.Temp[3,len(t1)-1] #daily
        me0=self.me/len(self.tf) #hourly
        #print(me, me0)
        efficacité=round(self.efficiency,2)
        print(10*"   ","\033[1m\033[4mSOLAR STILL PERFORMANCE AND TEMPERATURE PROFILES\033[0m\033[0m")
        print("Efficiency (absorbed energy) =",round(self.efficiency,2),"% | Efficiency (incident energy) =",round(self.efficiency_incident,2),"%")
        #Temperature in °C
        self.Temp[0:3,:]-=self.kelv
        #Temp[2,:]=(Temp[2,:]+((Tiex+Tin)/2 - self.kelv))/2
        self.cost_standard=self.cpk_standard(self.me)
        self.cost=self.cpk(self.me)
        self.yield_m2=self.me/self.ab            # daily yield per m2 of basin (kg/m2)
        self.invest_m2=self.usprice/self.ab      # initial investment per m2 of basin (USD/m2)
        print("Cost per kg:",round(self.cost,4),"USD (interest",self.interest,", lifetimes of the material sheet) |",round(self.cost_standard,4),"USD (12 % interest, 10 years)")
        print("Per m2 of basin: yield",round(self.yield_m2,2),"kg/m2/day | investment",round(self.invest_m2,1),"USD/m2")
        lisme=[self.device+"\n"+self.geo.dim[14], str(round(self.me,3))+" l/d", str(efficacité) +" "+"%",str(self.usprice)+" US", str(round(self.cost,4))+" "+"US/kg"]
        lisT=[  max(self.ig),max(self.ta)-self.kelv, max(self.Temp[0,:]),max(self.Temp[1,:]),max(self.Temp[2,:])]
        lisT1=[ np.mean(self.ig),np.mean(self.ta)-self.kelv,np.mean(self.Temp[0,:]),np.mean(self.Temp[1,:]),np.mean(self.Temp[2,:])]
        lisT2=[  min(self.ig),min(self.ta)-self.kelv, min(self.Temp[0,:]),min(self.Temp[1,:]),min(self.Temp[2,:])]
        lisT = [round(element, 2) for element in lisT]
        lisT1 = [round(element, 2) for element in lisT1]
        lisT2 = [round(element, 2) for element in lisT2]
        col=['Performance', 'Max','Mean', 'Min' ]
        effic = pd.DataFrame({col[0]: lisme},index=['Type', 'Daily prod', 'Efficiency','initial_invest', 'CPK'])
        Tprofil= pd.DataFrame({col[1]: lisT, col[2]: lisT1,col[3]: lisT2},index=[ 'I','Ta', 'Tg','Tw','Tb'])

        #Plot Temperature and distilled water profil
        plt.figure(figsize=(10,6))
        plt.subplot(3,3,1)
        plt.plot(t1,self.Temp[0,:], label='Tg',c='b')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,2)
        plt.plot(t1,self.Temp[1,:],label='Tw',c='r')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,3)
        plt.plot(t1,self.Temp[2,:],label='Tb',c="g")
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,4)
        plt.plot(t1,self.Temp[3,:],label='md',c='y')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,5)
        plt.scatter(self.tf, self.ta-self.kelv,c="c")
        plt.plot(t1, self.tamb(t1)-self.kelv,label='Ta', c="c")
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,6)
        plt.scatter(self.tf, self.ig,c="m")
        plt.plot(t1, self.ig1(t1),label='Ig',c="m")
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3,3,7)
        plt.plot(t1, (self.Tin- self.kelv),label='Ti',c="black")
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.grid()
        plt.subplot(3, 3, 8)
        plt.axis('off')
        plt.table(cellText=effic.values,rowLabels=effic.index ,colLabels=effic.columns,loc='bottom',cellColours= [['white'],['yellow'],['green'],['blue'],['red']], bbox=[0.3, 0, 0.7, 1], cellLoc='center')
        plt.subplot(3, 3, 9)
        plt.axis('off')
        plt.table(cellText=Tprofil.values,rowLabels=Tprofil.index ,colLabels=Tprofil.columns,loc='bottom',bbox=[0, 0, 1, 1], cellLoc='center')

        plt.show()

    def exp_data(self):
        self.exdata=read_sheet(self.excel_path, 'Exp_Data')
        self.read_excel_data()
        self.t=self.exdata.t_exp
        self.Texp=np.zeros([len(self.t),4])
        self.Texp[:,0]=self.exdata.Tg
        self.Texp[:,1]=self.exdata.Tw
        self.Texp[:,2]=self.exdata.Tb
        self.Texp[:,3]=self.exdata.md_cum
        efficiency0=0
        efficiency1=0
        for i in range(len(self.t)-1):
            efficiency0+=((self.Texp[i+1,3]-self.Texp[i,3])*self.Lv(self.Texp[i+1,1]))
            efficiency1+=((self.ig[i])*self.ab*self.conv)
        self.exp_efficiency=100*efficiency0/(efficiency1)
        self.cpkexp=self.cpk(self.Texp[len(self.t)-1,3])
        print("Exp_efficiency=",self.exp_efficiency,"%","\n", "Exp_CPK=",self.cpkexp)

    def store_data(self, store=False):
        self.exp_data()
        store = input("Would you like to store data? (yes/no): ")
        if store.lower() == "yes":
            if self.geo.dim[14] == "With baffles":
                gd14 = "Baff"
            else:
                gd14 = "WB"
            if self.device == "Conventional":
                gd14 = ""

            # Prepare data based on condition
            if not math.isnan(self.Texp[(len(self.t) - 1), 3]):
                tab2 = [self.gps, self.month,self.device + " SS " + gd14, self.mater, self.ns, self.ab, self.teta1, self.teta2, self.Texp[len(self.t) - 1, 3], self.exp_efficiency / 100, self.cpkexp, np.mean(self.ta) - self.kelv, max(self.ta) - self.kelv, min(self.ta) - self.kelv, np.mean(self.ig), max(self.ig), min(self.ig), "exp"]
                tab1 = [self.gps, self.month,self.device + " SS " + gd14, self.mater, self.ns, self.ab, self.teta1, self.teta2, self.me, self.efficiency / 100, self.cost, np.mean(self.ta) - self.kelv, max(self.ta) - self.kelv, min(self.ta) - self.kelv, np.mean(self.ig), max(self.ig), min(self.ig),"num"]
                self.data = [tab1, tab2]
            else:
                tab1 = [self.gps, self.month,self.device + " SS " + gd14, self.mater, self.ns, self.ab, self.teta1, self.teta2, self.me, self.efficiency / 100, self.cost, np.mean(self.ta) - self.kelv, max(self.ta) - self.kelv, min(self.ta) - self.kelv, np.mean(self.ig), max(self.ig), min(self.ig),"num"]
                self.data = [tab1]

            # Column names and index for the DataFrame
            columns = ["Site", 'Month','Type', "mat", "ns", "Ab", "tetag", "tetas", 'daily yield', 'efficiency', 'Cost', 'Ta(mean)', 'Ta(max)', 'Ta(min)', 'Ig(mean)', 'Ig(max)', 'Ig(min)', "E/N"]
            self.data_df = pd.DataFrame(self.data, columns=columns)
            print("Data was stored successfully")

            # Append to existing Excel stored
            with pd.ExcelWriter("stored.xlsx", engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                self.data_df.to_excel(writer, sheet_name="Data_ML", index=False, header=False, startrow=writer.sheets["Data_ML"].max_row)

    def online_storing(self):
        try:
            import gspread
            from google.colab import auth
            from google.auth import default
            # Authenticate for Google Sheets
            auth.authenticate_user()
            creds, _ = default()
            gc = gspread.authorize(creds)

            # Replace with your Google Sheet ID
            spreadsheet_id = '1oqn824PmTeq8ngPv5g9ny8LThceDqNDQLOeWLVMKrTE'
            sheet = gc.open_by_key(spreadsheet_id)

            # Specify worksheet
            worksheet = sheet.worksheet('Data_ML')

            # Convert data to a list of lists for Google Sheets
            data = [list(row) for row in self.data_df.itertuples(index=False, name=None)]

            # Find the next available row
            next_row = len(worksheet.get_all_values()) + 1

            # Update the sheet with new data starting from the next available row
            worksheet.update('A' + str(next_row), data)

            print("Data stored successfully in Google Sheet.")

        except Exception as e:
            print("An error occurred:", e)
      #############################################################################################################################
        #############################################################################################################################
###########################
class Error_deviation:
    #Error deviation initialization
    def error_calculation(self, ss, Temp=None):
        if Temp is None:
            Temp=ss.Temp
        excel_path=ss.excel_path
        ss.exp_data()
        if not math.isnan(ss.Texp[(len(ss.t) - 1), 3]):
            self.excel_path = excel_path
            exdata=read_sheet(ss.excel_path, 'Exp_Data')
            t=exdata.t_exp
            t2 = np.array([ss.t_flt(dt) for dt in t])
            inst=0
            inst_er=exdata.Exp_err_T[0]
            self.Texp=ss.Texp
            ss.read_excel_data()

            self.Temp=ss.Temp
            t1 = ss.simulate()
            #instrumental error
            for i in range (0,3):
                inst=inst+np.mean(self.Texp[:,i])**2
            inst=inst_er*np.sqrt(inst)/np.mean(self.Texp)
            # print("instrumental error=",100*round(inst,3), "%")
            # Initialize an array to store interpolated experimental data
            interpolated_funcs = []
            for i in range(self.Texp.shape[1]):  # loop over each column

                y2 = np.array(self.Texp[:, i]).reshape(len(self.Texp[:, i]),)
                # Create the interpolation function
                T_exp_interp = interp1d(t2, y2, kind='linear', fill_value='extrapolate')
                interpolated_funcs.append(T_exp_interp)

            T_exp_interpolated = np.zeros([len(t1), 4])
            for i in range(self.Texp.shape[1]):
                T_exp_interpolated[:, i] = interpolated_funcs[i](t1)  # Interpolate values

            #print(T_exp_interpolated)

            squared_diff=((np.mean(T_exp_interpolated.T[0:3,:])-np.mean(Temp[0:3,:]))/np.mean(T_exp_interpolated.T[0:3,:]))**2

            Err= (T_exp_interpolated.T[0:3,:]-(Temp[0:3,:]))/T_exp_interpolated.T[0:3,:]
            # Compute the Mean Squared Error (MSE) for each element
            mse = np.sqrt(squared_diff+inst**2)

            print("Mean-bias error (difference of the daily means) :", (round(100*mse,3)), "%")
            # Point-by-point errors: deviations do not cancel out
            names=['Tg','Tw','Tb']
            diff=Temp[0:3,:]-T_exp_interpolated.T[0:3,:]
            self.rmse=np.sqrt(np.mean(diff**2,axis=1))
            self.mape=100*np.mean(np.abs(diff)/np.abs(T_exp_interpolated.T[0:3,:]),axis=1)
            for k in range(3):
                print("  ",names[k],": RMSE =",round(self.rmse[k],2),"°C | mean absolute error =",round(self.mape[k],1),"%")
            md_exp=self.Texp[len(ss.t)-1,3]
            self.yield_error=100*(ss.me-md_exp)/md_exp
            print("   Daily yield : simulated",round(ss.me,3),"kg | measured",round(md_exp,3),"kg | deviation",round(self.yield_error,1),"%")
            self.bias_error=100*mse
            if 100*mse > 20:
                raise NotImplementedError("Incoming out of bound try to choose another correlation or verify input data")
            #Print error
            ##############################


            Err = (T_exp_interpolated.T[0:3, :] - Temp[0:3, :]) / T_exp_interpolated.T[0:3, :]

            # Convert Err matrix to DataFrame
            # Assuming 'Time' is the time array that corresponds to numerical and experimental data (tf or t1)
            df_err = pd.DataFrame({
                'Time': t1,  # Use t1 or any time array that fits the length of your data
                'Err Tg': Err[0, :],  # Error for cover
                'Err Tw': Err[1, :],  # Error for water
                'Err Tb': Err[2, :]   # Error for basin
            })

            # Add the instrumental error column
            df_err['Inst Error'] = inst

            # Plot Error values
            g = sns.lineplot(data=df_err, x="Time", y="Inst Error", color="r", linestyle=":")
            sns.lineplot(data=df_err, x="Time", y="Err Tg", ax=g)
            sns.lineplot(data=df_err, x="Time", y="Err Tw", ax=g)
            sns.lineplot(data=df_err, x="Time", y="Err Tb", ax=g)
            plt.xlabel('Time (h)', fontsize=11)
            plt.ylabel('Error', fontsize=11)
            plt.grid(True)
            plt.show()
        else:
            print("Experimental data are not available")


if __name__ == "__main__":
    # itr = number of time steps per measurement interval.
    # Results are converged from itr = 12; itr = 6 is too coarse for the conventional still.
    ss = SolarStill("conventional.xlsx", 24)
    ss.plot_device()
    ss.ss_performances()
    ss.plot_performances()
    Error_deviation().error_calculation(ss)
    # ss.store_data()       # append the results to stored.xlsx
    # ss.online_storing()   # Google Colab only: append the results to the shared Google Sheet
