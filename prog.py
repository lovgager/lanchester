from PyQt5.QtWidgets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class My_Ui:
    
    def setup_ui(self, w):
        self.w = w
        w.setWindowTitle('Lanchester')
        w.resize(1440, 720)

        self.figure = Figure(figsize=(80, 60))
        self.fc = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(121)
        self.ax_phase = self.figure.add_subplot(122)
        plt.subplots_adjust(left=0.03, right=0.98, wspace=0.25)
        self.init_axes()

        self.init_params()

        self.outer_layout = QGridLayout()
        buttons_layout = QVBoxLayout()

        self.outer_layout.addWidget(self.fc, 0, 1, 0, 8)
        self.outer_layout.addLayout(buttons_layout, 0, 0)

        button_analytical = QPushButton('Build Analytic')
        buttons_layout.addWidget(button_analytical)
        button_analytical.clicked.connect(self.build_an)

        button_numeric = QPushButton('Build Numeric')
        buttons_layout.addWidget(button_numeric)
        button_numeric.clicked.connect(self.build_num)

        button_clear = QPushButton('Clear')
        buttons_layout.addWidget(button_clear)
        button_clear.clicked.connect(self.clear)

        self.radio_euler = QRadioButton('Euler')
        self.radio_rk2 = QRadioButton('Runge-Kutta 2')
        self.radio_rk4 = QRadioButton('Runge-Kutta 4')
        self.radio_euler.setChecked(True)
        buttons_layout.addWidget(self.radio_euler)
        buttons_layout.addWidget(self.radio_rk2)
        buttons_layout.addWidget(self.radio_rk4)

        N_layout = QHBoxLayout()
        buttons_layout.addLayout(N_layout)
        N_layout.addWidget(QLabel('N = '))
        self.N_spinbox = QDoubleSpinBox()
        self.N_spinbox.setDecimals(0)
        self.N_spinbox.setRange(1, 999999)
        self.N_spinbox.setValue(self.N_tau)
        N_layout.addWidget(self.N_spinbox)

        tau_layout = QHBoxLayout()
        buttons_layout.addLayout(tau_layout)
        tau_layout.addWidget(QLabel('tau = '))
        self.tau_spinbox = QDoubleSpinBox()
        self.tau_spinbox.setDecimals(3)
        self.tau_spinbox.setValue(self.tau)
        tau_layout.addWidget(self.tau_spinbox)

        T_final_layout = QHBoxLayout()
        buttons_layout.addLayout(T_final_layout)
        T_final_layout.addWidget(QLabel('T_final = '))
        self.T_final_line = QLineEdit()
        self.T_final_line.setText(str(round(self.T_final, 2)))
        self.T_final_line.setReadOnly(True)
        T_final_layout.addWidget(self.T_final_line)

        alpha_layout = QHBoxLayout()
        buttons_layout.addLayout(alpha_layout)
        alpha_label = QLabel('Λ1 = ')
        alpha_label.setToolTip('(Скорострельность) x (вероятность поражения) первой стороны')
        alpha_layout.addWidget(alpha_label)
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setDecimals(3)
        self.alpha_spinbox.setValue(self.a)
        self.alpha_spinbox.setToolTip('(Скорострельность) x (вероятность поражения) первой стороны')
        alpha_layout.addWidget(self.alpha_spinbox)

        beta_layout = QHBoxLayout()
        buttons_layout.addLayout(beta_layout)
        beta_label = QLabel('Λ2 = ')
        beta_label.setToolTip('(Скорострельность) x (вероятность поражения) второй стороны')
        beta_layout.addWidget(beta_label)
        self.beta_spinbox = QDoubleSpinBox()
        self.beta_spinbox.setDecimals(3)
        self.beta_spinbox.setValue(self.b)
        self.beta_spinbox.setToolTip('(Скорострельность) x (вероятность поражения) второй стороны')
        beta_layout.addWidget(self.beta_spinbox)

        u0_layout = QHBoxLayout()
        buttons_layout.addLayout(u0_layout)
        u0_label = QLabel('N1_0 = ')
        u0_label.setToolTip('Начальная численность первой стороны')
        u0_layout.addWidget(u0_label)
        self.u0_spinbox = QDoubleSpinBox()
        self.u0_spinbox.setDecimals(0)
        self.u0_spinbox.setRange(1, 999999)
        self.u0_spinbox.setValue(self.u0)
        self.u0_spinbox.setToolTip('Начальная численность первой стороны')
        u0_layout.addWidget(self.u0_spinbox)

        v0_layout = QHBoxLayout()
        buttons_layout.addLayout(v0_layout)
        v0_label = QLabel('N2_0 = ')
        v0_label.setToolTip('Начальная численность второй стороны')
        v0_layout.addWidget(v0_label)
        self.v0_spinbox = QDoubleSpinBox()
        self.v0_spinbox.setDecimals(0)
        self.v0_spinbox.setRange(1, 999999)
        self.v0_spinbox.setValue(self.v0)
        self.v0_spinbox.setToolTip('Начальная численность второй стороны')
        v0_layout.addWidget(self.v0_spinbox)

        presets_layout = QHBoxLayout()
        buttons_layout.addLayout(presets_layout)
        presets_layout.addWidget(QLabel('Preset: '))
        self.presets = QComboBox()
        self.presets.addItem('first')
        self.presets.addItem('second')
        self.presets.addItem('draw')
        self.presets.addItem('draw2')
        self.presets.currentTextChanged.connect(self.preset_changed)
        presets_layout.addWidget(self.presets)

        reinforce_layout = QHBoxLayout()
        buttons_layout.addLayout(reinforce_layout)
        reinforce_layout.addWidget(QLabel('Reinforce: '))
        self.reinforce = QCheckBox()
        reinforce_layout.addWidget(self.reinforce)

    def init_params(self):
        self.u0 = 100
        self.v0 = 100
        self.a = 0.04
        self.b = 0.01
        self.tau = 0.1
        self.N_tau = 100
        self.calc_T_final()
        self.an_flag = False
        self.num_flag = False

    def calc_T_final(self):
        x = self.u0*np.sqrt(self.a) / (self.v0*np.sqrt(self.b))
        self.T_final = np.inf
        if x > 1:
            x = 1/x
        if np.abs(1-x) > 1e-10:
            self.T_final = 1/(2*np.sqrt(self.a*self.b)) * np.log((1+x)/(1-x))

    def init_axes(self):
        self.ax.set_xlabel('time')
        self.ax.set_title('N1(t), N2(t)')
        self.ax_phase.set_title('Phase plane')
        self.ax_phase.set_xlabel('N1')
        self.ax_phase.set_ylabel('N2')
        self.ax.grid()
        self.ax_phase.grid()

    def preset_changed(self, value):
        first = (100, 100, 0.04, 0.01)
        second = (100, 100, 0.01, 0.04)
        draw = (100, 100, 0.04, 0.04)
        draw2 = (100, 10, 0.01, 1)
        if value == 'first':
            self.u0, self.v0, self.a, self.b = first
        elif value == 'second':
            self.u0, self.v0, self.a, self.b = second
        elif value == 'draw':
            self.u0, self.v0, self.a, self.b = draw
        elif value == 'draw2':
            self.u0, self.v0, self.a, self.b = draw2
        self.u0_spinbox.setValue(self.u0)
        self.v0_spinbox.setValue(self.v0)
        self.alpha_spinbox.setValue(self.a)
        self.beta_spinbox.setValue(self.b)

    def build_an(self):
        self.u0 = self.u0_spinbox.value()
        self.v0 = self.v0_spinbox.value()
        self.a = self.alpha_spinbox.value()
        self.b = self.beta_spinbox.value()
        self.tau = self.tau_spinbox.value()
        self.N_tau = int(self.N_spinbox.value())
        self.calc_T_final()
        self.T_final_line.setText(str(round(self.T_final, 2)))

        T = self.T_final
        if T == np.inf:
            T = 200
        t = np.linspace(0, T, 1000)

        u0 = self.u0
        v0 = self.v0
        a = self.a
        b = self.b
        u_an = 0.5*(u0*np.sqrt(a) - v0*np.sqrt(b))/np.sqrt(a) * np.exp(np.sqrt(a*b)*t) + \
            0.5*(u0*np.sqrt(a) + v0*np.sqrt(b))/np.sqrt(a) * np.exp(-np.sqrt(a*b)*t)
        v_an = 0.5*(u0*np.sqrt(a) - v0*np.sqrt(b))/np.sqrt(b) * np.exp(np.sqrt(a*b)*t)*(-1) + \
            0.5*(u0*np.sqrt(a) + v0*np.sqrt(b))/np.sqrt(b) * np.exp(-np.sqrt(a*b)*t)

        if self.reinforce.isChecked():
            C = np.abs(self.a*self.u0**2 - self.b*self.v0**2)
            self.v0 = round(np.sqrt(C/self.b))
            v0 = self.v0
            self.calc_T_final()
            T2 = self.T_final
            tt = np.linspace(0, T2, 1000)
            u_an2 = 0.5*(u0*np.sqrt(a) - v0*np.sqrt(b))/np.sqrt(a) * np.exp(np.sqrt(a*b)*tt) + \
                0.5*(u0*np.sqrt(a) + v0*np.sqrt(b))/np.sqrt(a) * np.exp(-np.sqrt(a*b)*tt)
            v_an2 = 0.5*(u0*np.sqrt(a) - v0*np.sqrt(b))/np.sqrt(b) * np.exp(np.sqrt(a*b)*tt)*(-1) + \
                0.5*(u0*np.sqrt(a) + v0*np.sqrt(b))/np.sqrt(b) * np.exp(-np.sqrt(a*b)*tt)
            u_an = np.concatenate([u_an, u_an2])
            v_an = np.concatenate([v_an, v_an2])
            t2 = np.linspace(T, T + T2, 1000)
            t = np.concatenate([t, t2])
            self.T_final_line.setText(str(round(T + T2, 2)))

        #if self.an_flag:
        #    self.clear()

        self.ax.plot(t[u_an > 0.01], u_an[u_an > 0.01], label='N1 analytic')
        self.ax.plot(t[v_an > 0.01], v_an[v_an > 0.01], label='N2 analytic')
        self.ax_phase.plot(u_an, v_an, label='analytic')
        self.ax.legend()
        self.ax_phase.legend()
        self.fc.draw()
        #self.an_flag = True

    def build_num(self):
        self.u0 = self.u0_spinbox.value()
        self.v0 = self.v0_spinbox.value()
        self.a = self.alpha_spinbox.value()
        self.b = self.beta_spinbox.value()
        self.tau = self.tau_spinbox.value()
        self.N_tau = int(self.N_spinbox.value())
        self.calc_T_final()
        self.T_final_line.setText(str(round(self.T_final, 2)))

        u0 = self.u0
        v0 = self.v0
        a = self.a
        b = self.b
        tau = self.tau
        N_tau = self.N_tau

        T = tau*N_tau
        if T > self.T_final:
            T = self.T_final
            self.tau = T/N_tau
            tau = self.tau
            self.tau_spinbox.setValue(tau)
        t = np.linspace(0, T, N_tau + 1)

        y = np.zeros((N_tau + 1, 2))
        y[0] = [u0, v0]

        f = lambda y: np.array([-b*y[1], -a*y[0]])
        if self.radio_euler.isChecked():
            for n in range(N_tau):
                y[n + 1] = y[n] + tau*f(y[n])
                if self.reinforce.isChecked() and y[n + 1][0] <= 0.1*u0:
                    y[n + 1][0] = 1.1*u0
        elif self.radio_rk2.isChecked():
            for n in range(N_tau):
                y_predict = y[n] + tau*f(y[n])
                y[n + 1] = y[n] + tau/2*(f(y[n]) + f(y_predict))
        else:
            for n in range(N_tau):
                k1 = f(y[n])
                k2 = f(y[n] + tau/2*k1)
                k3 = f(y[n] + tau/2*k2)
                k4 = f(y[n] + tau*k3)
                y[n + 1] = y[n] + tau/6*(k1+2*k2+2*k3+k4)

        #if self.num_flag:
        #    self.clear()

        method = 'Euler'
        if self.radio_rk2.isChecked():
            method = 'RK2'
        elif self.radio_rk4.isChecked():
            method = 'RK4'
        self.ax.plot(t, y[:,0], label='N1 numeric '+method)
        self.ax.plot(t, y[:,1], label='N2 numeric '+method)
        self.ax_phase.plot(y[:,0], y[:,1], label='numeric '+method)
        self.ax.legend()
        self.ax_phase.legend()
        self.fc.draw()
        #self.num_flag = True

    def clear(self):
        self.ax.cla()
        self.ax_phase.cla()
        self.init_axes()
        self.fc.draw()
        #self.an_flag = False
        #self.num_flag = False


#===================================================================
#===================================================================

if __name__ == "__main__":
    app = QApplication([])
    w = QWidget()
    ui = My_Ui()
    ui.setup_ui(w)

    w.setLayout(ui.outer_layout)
    w.show()
    app.exec_()