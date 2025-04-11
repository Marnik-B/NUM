import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 300
alpha = 0.7
beta = 0.4
dt = 0.1

# Fonctions du système
def F_S(S):
    return -alpha * S

def F_G(M):
    return beta * M

def F_M(S, M):
    return alpha* S - beta * M

# ----------------- Euler method -----------------
def euler():
    M = np.array([1])
    S = np.array([N - M[0]])
    G = np.array([0])
    t = np.array([0])

    while M[-1] >= 0.01:
        Sn, Mn, Gn = S[-1], M[-1], G[-1]

        S_next = Sn + dt * F_S(Sn)
        G_next = Gn + dt * F_G(Mn)
        M_next = Mn + dt * F_M(Sn, Mn)
        t_next = t[-1] + dt

        S=np.append(S,S_next)
        G=np.append(G,G_next)
        M=np.append(M,M_next)
        t=np.append(t,t_next)

    return t, S, M, G

# ----------------- RK2 method -----------------
def rk2():
    M = np.array([1])
    S = np.array([N - M[0]])
    G = np.array([0])
    t = np.array([0])

    while M[-1] >= 0.01:
        Sn, Mn, Gn = S[-1], M[-1], G[-1]
        
        S_next = Sn + dt*F_S(Sn+dt/2*F_S(Sn))
        G_next = Gn + dt*F_G(Mn+dt/2*F_G(Mn))
        M_next = Mn + dt*F_M(Sn,Mn+dt/2*F_M(Sn,Mn))
        t_next = t[-1] + dt

        S=np.append(S,S_next)
        G=np.append(G,G_next)
        M=np.append(M,M_next)
        t=np.append(t,t_next)

    return t,S,M,G

t_e, S_e, M_e, G_e = euler()
t_rk, S_rk, M_rk, G_rk = rk2()

plt.figure(figsize=(10,6))
plt.plot(t_e, M_e, label="Malades (Euler)", linestyle="--")
plt.plot(t_rk, M_rk, label="Malades (RK2)", linestyle="-")
plt.title("Comparaison Euler vs RK2")
plt.xlabel("Temps (jours)")
plt.ylabel("Nombre de malades")
plt.legend()
plt.grid()
plt.show()

# Résultats
print("Résultats RK2:")
print(f"Pic de l'épidémie à t = {t_rk[np.argmax(M_rk)]} jours")
print(f"Fin de l'épidémie à t = {t_rk[-1]} jours")
print(f"Total infectés: {int(G_rk[-1] + M_rk[-1])}")

print("\nRésultats Euler:")
print(f"Pic de l'épidémie à t = {t_e[np.argmax(M_e)]} jours")
print(f"Fin de l'épidémie à t = {t_e[-1]} jours")
print(f"Total infectés: {int(G_e[-1] + M_e[-1])}")

