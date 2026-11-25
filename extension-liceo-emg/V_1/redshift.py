import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# 1. Redshift y Factor de Escala
# ==========================================
def redshift(lambda_rest, lambda_obs):
    """Calcula el corrimiento al rojo z a partir de longitudes de onda."""
    return (lambda_obs - lambda_rest) / lambda_rest

def scale_factor(z):
    """Calcula el factor de escala cosmológico a = 1 / (1 + z)."""
    return 1.0 / (1.0 + z)

# ==========================================
# 2. Parámetros del Modelo Cosmológico (Flat LCDM)
# ==========================================
c = 299792.458       # Velocidad de la luz en km/s
H0 = 70.0            # Constante de Hubble en km s^-1 Mpc^-1
Omega_M = 0.3        # Densidad de materia
Omega_L = 0.7        # Densidad de energía oscura (Lambda)
Omega_k = 0.0        # Curvatura espacial (Universo plano)

dH = c / H0          # Distancia de Hubble en Mpc (~4282.75 Mpc)

def E(z, omega_m=Omega_M, omega_l=Omega_L, omega_k=Omega_k):
    """Función de expansión normalizada E(z) = H(z) / H0."""
    return np.sqrt(omega_m * (1.0 + z)**3 + omega_k * (1.0 + z)**2 + omega_l)

# ==========================================
# 3. Distancia Comóvil y de Luminosidad
# ==========================================
def comoving_distance_single(z):
    """Calcula la distancia comóvil dC(z) para un único valor de z."""
    integrand = lambda zp: 1.0 / E(zp)
    integral, _ = quad(integrand, 0.0, z)
    return dH * integral

# Vectorizamos la función para evaluar arrays de numpy cómodamente
comoving_distance = np.vectorize(comoving_distance_single)

def luminosity_distance(z):
    """Calcula la distancia de luminosidad dL(z) = (1 + z) * dC(z)."""
    return (1.0 + z) * comoving_distance(z)

# ==========================================
# Datos de Entrada y Cálculos
# ==========================================
lambda_rest = 6500.0  # Å (por ejemplo, línea H-alfa en reposo ~6563 Å o referencia 6500 Å)
lambda_obs = np.linspace(6500.0, 20000.0, 200)  # Rango continuo de longitudes de onda observadas en Å

# Cálculo de magnitudes cosmológicas
z_vals = redshift(lambda_rest, lambda_obs)
a_vals = scale_factor(z_vals)
dL_vals = luminosity_distance(z_vals)

# ==========================================
# 4. Visualización Comparativa
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

# Panel 1: Lambda_obs vs Redshift (z)
axes[0].plot(z_vals, lambda_obs, color='firebrick', lw=2)
axes[0].set_xlabel("Redshift ($z$)", fontsize=11)
axes[0].set_ylabel(r"$\lambda_{\rm obs}$ ($\mathrm{\AA}$)", fontsize=11)
axes[0].set_title(r"$\lambda_{\rm obs}$ vs. Redshift ($z$)", fontsize=12)
axes[0].grid(True, linestyle="--", alpha=0.6)

# Panel 2: Lambda_obs vs Factor de Escala (a)
axes[1].plot(a_vals, lambda_obs, color='teal', lw=2)
axes[1].set_xlabel("Factor de escala ($a$)", fontsize=11)
axes[1].set_ylabel(r"$\lambda_{\rm obs}$ ($\mathrm{\AA}$)", fontsize=11)
axes[1].set_title(r"$\lambda_{\rm obs}$ vs. Factor de Escala ($a$)", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.6)

# Panel 3: Lambda_obs vs Distancia de Luminosidad (dL)
axes[2].plot(dL_vals, lambda_obs, color='darkslateblue', lw=2)
axes[2].set_xlabel("Distancia de luminosidad $d_L$ (Mpc)", fontsize=11)
axes[2].set_ylabel(r"$\lambda_{\rm obs}$ ($\mathrm{\AA}$)", fontsize=11)
axes[2].set_title(r"$\lambda_{\rm obs}$ vs. $d_L(z)$", fontsize=12)
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.suptitle(
    rf"Expansión Cosmológica $\Lambda\mathrm{{CDM}}$ ($H_0={H0}$, $\Omega_M={Omega_M}$, $\Omega_\Lambda={Omega_L}$)",
    fontsize=13, fontweight='bold'
)

plt.show()


#### Este es un código para una clase para niños de 15 años: ¿Qué es el redshift?
# ==============================================================================
# Ejemplo 1: Espectro de una galaxia en REPOSO (z = 0)
# Imaginemos la luz de la galaxia como un continuo (nivel base) con "picos" de luz brillante (líneas de emisión)

# 1. Rango de longitud de onda (en Angstroms) que podemos ver
longitud_onda = np.linspace(4000, 10000, 1000)  # De 4000 Å a 10000 Å

# 2. Luz base de la galaxia (el continuo de estrellas)
continuo = 1.0  # Nivel base simple y constante

# 3. Líneas de emisión en reposo (como su "código de barras" químico de laboratorio)
O_III_reposo = 5007   # Oxígeno ionizado ([O III]) en color verde-azulado (~5007 Å)
H_alfa_reposo = 6563  # Hidrógeno alfa (H-alfa) en color rojo (~6563 Å)

# 4. Creamos los "picos" de brillo (funciones Gaussianas sencillas)
pico_O_III = 1.8 * np.exp(-0.5 * ((longitud_onda - O_III_reposo) / 30)**2)
pico_H_alfa = 2.5 * np.exp(-0.5 * ((longitud_onda - H_alfa_reposo) / 30)**2)

# Espectro total = luz base + picos de emisión
flujo_total = continuo + pico_O_III + pico_H_alfa

# 5. Gráfico muy visual y claro para explicar a los alumnos
plt.figure(figsize=(10, 5))
plt.plot(longitud_onda, flujo_total, color='black', lw=2, label="Espectro de la galaxia (en reposo, z = 0)")

# Marcamos y destacamos dónde están las líneas
plt.axvline(O_III_reposo, color='teal', linestyle='--', lw=1.5, alpha=0.8)
plt.text(O_III_reposo + 60, 2.7, r"[O III] (5007 $\mathrm{\AA}$)", color='teal', fontsize=11, fontweight='bold')

plt.axvline(H_alfa_reposo, color='crimson', linestyle='--', lw=1.5, alpha=0.8)
plt.text(H_alfa_reposo + 60, 3.4, r"H$\alpha$ (6563 $\mathrm{\AA}$)", color='crimson', fontsize=11, fontweight='bold')

# Detalles del gráfico
plt.title("Luz de una galaxia en reposo (en el laboratorio)", fontsize=13, fontweight='bold')
plt.xlabel(r"Longitud de onda $\lambda$ ($\mathrm{\AA}$)", fontsize=11)
plt.ylabel("Brillo / Intensidad de luz", fontsize=11)
plt.xlim(4000, 10000)
plt.ylim(0, 4.0)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)

plt.show()

# ==============================================================================
# Ejemplo 2: La misma galaxia observada a REDSHIFT z = 2
# La luz viaja por el Universo en expansión y su longitud de onda se estira:
# lambda_obs = lambda_reposo * (1 + z) = lambda_reposo * 3

z_clase = 2.0
factor_estiramiento = 1.0 + z_clase  # 1 + 2 = 3

# Nuevas posiciones observadas de las líneas de emisión:
O_III_obs = O_III_reposo * factor_estiramiento    # 5007 * 3 = 15021 Å (Infrarrojo)
H_alfa_obs = H_alfa_reposo * factor_estiramiento  # 6563 * 3 = 19689 Å (Infrarrojo)

# Rango de observación más amplio para poder verlas (alcanzan el infrarrojo cercano)
longitud_onda_obs = np.linspace(4000, 24000, 2000)

# El ancho de las líneas también se estira por el mismo factor (1+z)
ancho_linea_obs = 30.0 * factor_estiramiento

# Picos desplazados
pico_O_III_obs = 1.8 * np.exp(-0.5 * ((longitud_onda_obs - O_III_obs) / ancho_linea_obs)**2)
pico_H_alfa_obs = 2.5 * np.exp(-0.5 * ((longitud_onda_obs - H_alfa_obs) / ancho_linea_obs)**2)

flujo_observado = continuo + pico_O_III_obs + pico_H_alfa_obs

# Gráfico del espectro corrido al rojo
plt.figure(figsize=(12, 5))
plt.plot(longitud_onda_obs, flujo_observado, color='darkorange', lw=2, 
         label=f"Espectro observado a $z = {z_clase}$ (estirado $\\times {factor_estiramiento:.0f}$)")

# Marcamos dónde estaban ORIGINALMENTE en el laboratorio (reposo) como referencia fantasma
plt.axvline(O_III_reposo, color='teal', linestyle=':', lw=1.2, alpha=0.5)
plt.axvline(H_alfa_reposo, color='crimson', linestyle=':', lw=1.2, alpha=0.5)
plt.text(5200, 0.4, "Posiciones originales\n(en reposo)", color='gray', fontsize=9, ha='center')

# Flechas que muestran el desplazamiento ("el estiramiento")
plt.annotate('', xy=(O_III_obs, 2.3), xytext=(O_III_reposo, 2.3),
             arrowprops=dict(arrowstyle="->", color="teal", lw=1.5, ls="--"))
plt.annotate('', xy=(H_alfa_obs, 2.9), xytext=(H_alfa_reposo, 2.9),
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5, ls="--"))

# Nuevas posiciones observadas
plt.axvline(O_III_obs, color='teal', linestyle='--', lw=1.5)
plt.text(O_III_obs + 150, 2.6, rf"[O III] observado ({O_III_obs:.0f} $\mathrm{{\AA}}$)", 
         color='teal', fontsize=10, fontweight='bold')

plt.axvline(H_alfa_obs, color='crimson', linestyle='--', lw=1.5)
plt.text(H_alfa_obs + 150, 3.2, rf"H$\alpha$ observado ({H_alfa_obs:.0f} $\mathrm{{\AA}}$)", 
         color='crimson', fontsize=10, fontweight='bold')

# Detalles pedagógicos
plt.title(rf"Galaxia lejana observada a $z = 2$ ($\lambda_{{\rm obs}} = 3 \times \lambda_{{\rm reposo}}$)", 
          fontsize=13, fontweight='bold')
plt.xlabel(r"Longitud de onda observada $\lambda_{\rm obs}$ ($\mathrm{\AA}$)", fontsize=11)
plt.ylabel("Brillo / Intensidad de luz", fontsize=11)
plt.xlim(4000, 24000)
plt.ylim(0, 4.0)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)

plt.show()


