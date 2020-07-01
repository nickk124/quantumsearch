#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:07:22 2020

@author: ericyelton
"""

#Imports
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit,execute, Aer

# gates/operators
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import random_unitary

#visualization
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation



#in this algorithm we have a dimension of n for the state space
#then we have log2(n) states in the coin space

#defining the dimensions of the quantum circuit
class QRWreg:
    """
    QRWreg initializes a Quantum Random Walk registry

    it inputs the variable dim which is the number of qubits in

    the state space

    """
    #dim is the number of qubits
    #c is the number of coin qubits
    def __init__(self,dim):
        c = (np.log2(dim))
        if c.is_integer() != 1:
            raise TypeError("The number of qubits n must have log2(n) be an integer")
        self.c = int(c)
        self.dim = dim
        n = int(dim+c)
        #the total number of qubits used will be given by
        self.n = n
        qr = QuantumRegister(n,'q')
        cl = ClassicalRegister(n,"c")
        self.qr = qr
        self.cl = cl
        circ = QuantumCircuit(qr, cl)
        self.circuit = circ


#defining the QRW search algorithm as a subclass of the QRW registry
class QRWsearch(QRWreg):
    """
    QRWsearch is an implimentation of the quantum random walk search algorithm
    based on the paper titled "Quantum random-walk search algorithm" from
    N. Shenvi et al
    this class ineherits the QRWreg class
    dim - number of qubits in the state space
    state_search - the state that the algorithm searches for

    """
    def __init__(self,dim,state_search):
        QRWreg.__init__(self,dim)
        self.circuit = QRWreg(dim).circuit
        quan_reg = QRWreg(dim).qr
        clas_reg = QRWreg(dim).cl
        self.search = state_search

        circ = self.circuit
        #hadamards on all of the states
        circ.h(quan_reg[0:self.n])

        #operating U prime pi/2 sqrt(2) times
        times = np.ceil(0.5*np.pi*np.sqrt(2**(self.dim)))


        for i in range(int(times)):
            circ.unitary(self.Uprime(),quan_reg,label="U'")

        #measure the registry onto the classical bits
        circ.measure(quan_reg,clas_reg)


    #defining all of the operators for the QRW search
    def S(self, display_matrix=False):
        coin = self.c
        num_dir = int(2 ** coin)
        state = self.dim
        S = np.zeros((2**(coin+state),2**(coin+state)))
        for i in range(num_dir):
            for j in range(2**state):
                #performing the bit flip using an XOR
                j_bin = int(j)
                e_i = int(i+1)
                xor = j_bin ^ e_i
                row = xor+(i*(2**state))
                S[row][j+(i*(2**state))] =1
        if display_matrix:
            print('Matrix for the shift operator:')
            print(S)
        return Operator(S)

    def C(self, display_matrix=False):
        #definition of the C operator is an outer product in the coin space
        #tensor product with the identity in state spac
        coin = self.c
        num_dir = int(2 ** coin)
        state = self.dim
        num_state = int(2**state)

        #defining the operator in just the coin space
        s_c = np.zeros((2**(coin),2**(coin)))
        I_c = np.zeros((2**(coin),2**(coin)))

        for i in range(num_dir):
            I_c[i][i] = 1
            for j in range(num_dir):
                s_c[i][j] =num_dir**-1
        s_c = 2*s_c
        s_c = Operator(s_c)
        I_c = Operator(I_c)
        G = s_c-I_c

        #defining the identity in the state space
        I_s = np.zeros((2**(state),2**(state)))
        for i in range(num_state):
            I_s[i][i] = 1
        I = Operator(I_s)
        C = G.tensor(I)
        if display_matrix:
            print('Matrix for the quantum coin operator:')
            print(np.real(C.data))
        return C

    def U(self, display_matrix=False):

        S_= self.S()
        C_ = self.C()

        U = C_.compose(S_)
        if display_matrix:
            print('Matrix for U:')
            print(np.real(U.data))

        return U

    def Uprime(self):
        #state_search is the state we are searching for
        #we will focus on the second term
        #Note that state search must be in decimal
        if self.search >= 2**self.dim:
            raise TypeError("Search State parameter is outside of state space values")
        elif self.search < 0:
            raise TypeError("Search State parameter is outside of state space values")
        else:
            #focusings on the second term of Uprime
            coin = self.c
            num_dir = int(2**coin)
            state = self.dim
            num_state = int(2**state)

            search_array = np.zeros((num_state,num_state))
            search_array[self.search][self.search] = 1
            search = Operator(search_array)

            s_c = np.zeros((2**(coin),2**(coin)))
            for i in range(num_dir):
                for j in range(num_dir):
                    s_c[i][j] = num_dir**-1

            coin_ = Operator(s_c)
            search_op = coin_.tensor(search)

            S_ = self.S()

            term2 = search_op.compose(S_)

            U_ = self.U()

            Uprime = U_-(2*term2)
            return Uprime

    #Visualization
    def draw_circuit(self):
        return self.circuit.draw(output='mpl')

    def plot_states_hist(self):  # plots by actually measuring the circuit
        #self.circuit.measure_all()
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024 # number of times circuit is run, for sampling
        results = execute(self.circuit, backend=backend, shots=shots).result()
        answer = results.get_counts()


        return plot_histogram(answer, figsize=(5,5))

#defining the QRW algorithm as a subclass of the QRW registry
class QRW(QRWreg):
    """
    The QRW Class is an arbitrary implementation of a QRW

    dim - number of qubits in the state space

    c_label = is a string that determines what coin operator to use on the coin space
        - Hadamard = Hadamard operator tensor product with the Identity
        -Random = Is a random unitary operator

    step - is the number of 'steps' the QRW completes it is the
            number of times the operator U is called

    """
    def __init__(self,dim,c_label,step):
        QRWreg.__init__(self,dim)
        self.circuit = QRWreg(dim).circuit
        quan_reg = QRWreg(dim).qr
        clas_reg = QRWreg(dim).cl
        circ = self.circuit

        #hadamards on all of the states
        circ.h(quan_reg[0:self.n])

        for i in range(int(step)):
            circ.unitary(self.U(c_label),quan_reg,label="Step")

        #measure the registry onto the classical bits
        circ.measure(quan_reg,clas_reg)


    #defining all of the operators for the QRW search
    def S(self):
        coin = self.c
        num_dir = int(2 ** coin)
        state = self.dim
        S = np.zeros((2**(coin+state),2**(coin+state)))
        for i in range(num_dir):
            for j in range(2**state):
                #performing the bit flip using an XOR
                j_bin = int(j)
                e_i = int(i+1)
                xor = j_bin ^ e_i
                row = xor+(i*(2**state))
                S[row][j+(i*(2**state))] =1


        return Operator(S)

    def C(self,c_label):
        coin = self.c
        state = self.dim

        #creating the identity in the S space
        I = np.zeros((2**state,2**state))
        for i in range(2**state):
            I[i][i] = 1
        I = Operator(I)

        if c_label == "Hadamard":
            result= np.zeros((2**coin,2**coin))
            for i in range(2**coin):
                for j in range(2**coin):
                    #bin_i = bin(i)
                    #bin_j = bin(j)
                    if i >= 2 and j >= 2:
                        result[i][j] = -1*(-1)**(i * j)*(2**(-1*(0.5*coin)))
                    else:
                        result[i][j] = (-1)**(i * j)*(2**(-1*(0.5*coin)))

            res_op = (Operator(result))
            C_final = res_op.tensor(I)

            return C_final

        elif c_label == "Random":
            dim = []
            for i in range(coin):
                dim.append(2)
            res_op = random_unitary(tuple(dim))
            C_final = res_op.tensor(I)
            return C_final
        else:
            raise TypeError("Label string for C is not a valid input")

    def U(self,c_label):

        S_= self.S()
        C_ = self.C(c_label)

        U = C_.compose(S_)
        return U

    #Visualization
    def draw_circuit(self):
        return self.circuit.draw()

    def plot_states_hist(self):  # plots by actually measuring the circuit

        backend = Aer.get_backend('qasm_simulator')
        print(backend)
        shots = 1024 # number of times circuit is run, for sampling
        results = execute(self.circuit, backend=backend, shots=shots).result()
        print(results.get_counts())
        answer = results.get_counts()

        return plot_histogram(answer, figsize=(15,5))

    #controlling the circuit
    def execute(self):
        backend = Aer.get_backend('qasm_simulator')
        results = execute(self.circuit, backend=backend,shots = 1).result()
        answer = results.get_counts()

        #one execution means there will be one state in the answer dict
        state = list(answer.keys())
        return state[0]

#animating the 2D 'mapping' for the QRW it inherits the QRW class
class QRW_Automata(QRW):
    """
    QRW_Automata is a class that inherits the QRW class
    it animates multiple executions of the QRW algorithm into a
    cellular automata board

    Note this algorithm has only been defined for the cases of 2 and 4 state qubits!

    dim - number of qubits in the state space

    c_label = is a string that determines what coin operator to use on the coin space
        - Hadamard = Hadamard operator tensor product with the Identity
        -Random = Is a random unitary operator

    step - is the number of 'steps' the QRW completes it is the
            number of times the operator U is called

    iters - number of times the circuit is executed (also determines the number of
                                                     frames in the animation)

    """

    def __init__(self,dim,c_label,step,iters):
        QRW.__init__(self,dim,c_label,step)
        self.n = QRW(dim,c_label,step).n
        self.c = QRW(dim,c_label,step).c
        self.circ = QRW(dim,c_label,step).circuit
        state = []
        for i in range(iters):
            state.append(QRW(dim,c_label,step).execute())

        self.state = state

        print("state")
        print(state)
        c_state = []
        s_state = []
        for i in state:
            c_state.append(i[0:self.c])
            s_state.append(i[self.c:])
        self.c_state = c_state
        self.s_state = s_state
        if (dim/2).is_integer() != True:
            raise ValueError("The one-half of the number of qubits in the state space must be an integer")

        #dividing up the board according to number of bits
        #n is the number of rows and columns
        #we are just doing the case for n = 4
        if int(self.dim) !=4 and int(self.dim) !=2:
            raise ValueError("Grid is implemented for only for the cases where n=2 and n=4 in the state space")

        figure,axes = plt.subplots(nrows = 2,ncols = 1)
        plt.title("QRW Automata {step}".format(step=c_label))
        axes[0].set_facecolor((0,0,0))
        axes[0].get_xaxis().set_ticks([])
        axes[0].get_yaxis().set_ticks([])
        axes[1].get_xaxis().set_ticks([])
        axes[1].get_yaxis().set_ticks([])

        self.fig = figure
        self.ax = axes
        self.circuit.draw('mpl',scale = 0.9,ax=axes[1])

        n_v = (2*self.dim)
        if self.c % 2 !=0:
            n_h = self.dim
        else: n_h = n_v
        v_lines = np.arange(0,1,1/n_v)
        h_lines = np.arange(0,1,1/n_h)

         #drawing the frames for the board
        for i in v_lines:
            axes[0].axvline(x=i,ymin=0,ymax=1,color='r')
        for i in h_lines:
            axes[0].axhline(y=i,xmin=0,xmax=1,color='r')

        anim = FuncAnimation(self.fig,self.animate,
                             frames =int(len(state)),
                             interval = 200)
        anim.save("QRW_{n}qubits_{op}_{st}steps.mp4".format(n = self.dim,op = c_label,st = step),)

    def animate(self,i):

        #ploting the state space :
        if self.dim ==2:

            c_ = self.c_state[i]
            s_ = self.s_state[i]
            n_v = (2*self.dim)
            n_h = self.dim
            verts = [
                (0.+int(s_[-1])*(1/n_v)+0.5*int(c_), 1.-int(s_[0])*(1/n_h)-(1/n_h)),  # left, bottom
                (0.+int(s_[-1])*(1/n_v)+0.5*int(c_), 1.-int(s_[0])*(1/n_h)),  # left, top
                ((1/n_v)+int(s_[-1])*(1/n_v)+0.5*int(c_), 1.-int(s_[0])*(1/n_h)),  # right, top
                ((1/n_v)+int(s_[-1])*(1/n_v)+0.5*int(c_), 1.-int(s_[0])*(1/n_h)-(1/n_h)),  # right, bottom
                (0.+int(s_[-1])*(1/n_v)+0.5*int(c_), 1.-int(s_[0])*(1/n_h)-(1/n_h)),  # ignored
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            path = Path(verts,codes)
            cell = patch.PathPatch(path,facecolor = 'w')
            self.ax[0].set_xlabel("Measured state:{state_}".format(state_=self.state[i]))
            self.ax[0].add_patch(cell)
            return patch
        elif self.dim ==4:
            c_ = self.c_state[i]
            s_ = self.s_state[i]
            n_v = (2*self.dim)
            n_h = (2*self.dim)
            verts = [
                (0.+int(s_[-1])*(1/n_v)+int(s_[-2])*(2/n_v)+0.5*int(c_[1]), 1.-int(s_[0])*2*(1/n_h)-int(s_[1])*(1/n_h)-(1/n_h)-0.5*int(c_[0])),  # left, bottom
                (0.+int(s_[-1])*(1/n_v)+int(s_[-2])*(2/n_v)+0.5*int(c_[1]), 1.-int(s_[0])*2*(1/n_h)-int(s_[1])*(1/n_h)-0.5*int(c_[0])),  # left, top
                ((1/n_v)+int(s_[-1])*(1/n_v)+int(s_[-2])*(2/n_v)+0.5*int(c_[1]), 1.-int(s_[0])*2*(1/n_h)-int(s_[1])*(1/n_h)-0.5*int(c_[0])),  # right, top
                ((1/n_v)+int(s_[-1])*(1/n_v)+int(s_[-2])*(2/n_v)+0.5*int(c_[1]), 1.-int(s_[0])*2*(1/n_h)-int(s_[1])*(1/n_h)-(1/n_h)-0.5*int(c_[0])),  # right, bottom
                (0.+int(s_[-1])*(1/n_v)+int(s_[-2])*(2/n_v)+0.5*int(c_[1]), 1.-int(s_[0])*2*(1/n_h)-int(s_[1])*(1/n_h)-(1/n_h)-0.5*int(c_[0])),  # ignored
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]

            path = Path(verts,codes)
            cell = patch.PathPatch(path,facecolor = 'w')
            self.ax[0].set_xlabel("Measured state:{state_}".format(state_=self.state[i]))
            self.ax[0].add_patch(cell)
            return patch

if __name__=='main': 
    QRW_Automata(4,"Hadamard",4,15)