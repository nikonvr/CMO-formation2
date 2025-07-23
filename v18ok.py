import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
jax.config.update("jax_enable_x64", False)
import matplotlib.pyplot as plt
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional
from scipy.optimize import minimize, OptimizeResult, differential_evolution
import time
import datetime
import traceback
from collections import deque

# --- SECTION : CONSTANTES ET CONFIGURATION ---

MIN_THICKNESS_PHYS_NM = 0.01
MAXITER_HARDCODED = 1000
MAXFUN_HARDCODED = 1000
MAX_LAYERS = 10
MATERIAL_KEYS = ['H', 'L', 'A', 'B', 'C']
DTYPE_COMPLEX = jnp.complex64
DTYPE_FLOAT = jnp.float32

# --- CONSTANTES POUR LA COLORIMÉTRIE ---
CIE_LAMBDA = np.arange(380, 781, 5)
CIE_X = np.array([
    0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776, 0.1344, 0.2148, 0.3230,
    0.4479, 0.5970, 0.7621, 0.9163, 1.0263, 1.0622, 1.0456, 1.0026, 0.9564, 0.9154, 0.8634,
    0.7889, 0.6954, 0.5945, 0.4900, 0.3856, 0.2899, 0.2091, 0.1484, 0.1041, 0.0734, 0.0514,
    0.0358, 0.0249, 0.0172, 0.0117, 0.0081, 0.0058, 0.0045, 0.0036, 0.0029, 0.0024, 0.0020,
    0.0017, 0.0014, 0.0011, 0.0009, 0.0007, 0.0005, 0.0004, 0.0003, 0.0002, 0.0002, 0.0001,
    0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000
])
CIE_Y = np.array([
    0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022, 0.0040, 0.0073, 0.0129,
    0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540,
    0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1170,
    0.0782, 0.0526, 0.0353, 0.0231, 0.0154, 0.0106, 0.0074, 0.0053, 0.0039, 0.0029, 0.0021,
    0.0016, 0.0012, 0.0008, 0.0006, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000
])
CIE_Z = np.array([
    0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713, 0.6456, 1.0391, 1.5281,
    2.0561, 2.5861, 3.0781, 3.4828, 3.7008, 3.6551, 3.4481, 3.1870, 2.9080, 2.6480, 2.3481,
    1.9961, 1.6361, 1.2880, 0.9693, 0.6934, 0.4692, 0.3162, 0.2120, 0.1419, 0.0954, 0.0640,
    0.0426, 0.0283, 0.0188, 0.0125, 0.0084, 0.0057, 0.0041, 0.0031, 0.0023, 0.0018, 0.0014,
    0.0011, 0.0009, 0.0006, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000
])
ILLUMINANT_D65 = np.array([
    35.8, 52.5, 66.3, 81.3, 101.5, 114.9, 129.8, 143.1, 145.4, 137.9, 131.1, 128.8, 126.9,
    128.1, 122.9, 115.8, 111.8, 108.6, 108.9, 108.8, 107.5, 106.3, 106.8, 106.1, 104.9,
    103.4, 103.1, 103.2, 101.4, 100.8, 100.0, 98.7, 96.6, 95.7, 92.0, 90.1, 89.1, 89.2,
    88.0, 85.1, 82.5, 81.4, 82.0, 81.6, 80.3, 77.2, 74.3, 72.8, 70.2, 67.5, 66.8, 66.1,
    64.1, 62.0, 61.3, 59.6, 57.7, 56.1, 54.1, 52.2, 51.5, 49.6, 47.6, 46.1, 45.3, 45.4,
    44.1, 42.0, 40.2, 38.6, 37.3, 35.5, 33.7, 32.2, 30.6, 29.3, 28.5, 27.8, 27.0, 26.2,
    25.4
])
# Calcul du point blanc de référence pour D65
Y_n_sum = np.sum(ILLUMINANT_D65 * CIE_Y)
k_n = 100.0 / Y_n_sum
Xn = k_n * np.sum(ILLUMINANT_D65 * CIE_X)
Yn = 100.0
Zn = k_n * np.sum(ILLUMINANT_D65 * CIE_Z)
XYZ_N_D65 = np.array([Xn, Yn, Zn])

# Matrice de transformation XYZ vers sRGB (pour illuminant D65)
XYZ_TO_RGB_MATRIX = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

# --- SECTION : FONCTIONS DE JOURNALISATION (LOG) ---

def add_log_message(message: str):
    """Ajoute un message horodaté au journal de l'application."""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = deque(maxlen=200)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.appendleft(f"[{timestamp}] {message}")

def add_log(messages: Union[str, List[str]]):
    """Wrapper pour ajouter un ou plusieurs messages au journal."""
    if isinstance(messages, str):
        messages = [messages]
    for msg in messages:
        if msg:
            add_log_message(msg)

# --- SECTION : FONCTIONS DE CALCUL COLORIMÉTRIQUE ---

def spectrum_to_xyz(l_reflect, reflect, l_cie, x_bar, y_bar, z_bar, illuminant):
    """Convertit un spectre de réflectance en coordonnées CIE XYZ."""
    reflect_resampled = np.interp(l_cie, l_reflect, reflect)
    
    X = np.sum(reflect_resampled * illuminant * x_bar)
    Y = np.sum(reflect_resampled * illuminant * y_bar)
    Z = np.sum(reflect_resampled * illuminant * z_bar)
    
    k = 100.0 / np.sum(illuminant * y_bar)
    return np.array([X, Y, Z]) * k

def xyz_to_lab(xyz, xyz_n):
    """Convertit les coordonnées CIE XYZ en CIE L*a*b*."""
    xyz_ref = xyz / xyz_n
    
    def f(t):
        delta = 6.0 / 29.0
        return np.where(t > delta**3,
                        t**(1.0/3.0),
                        (t / (3.0 * delta**2)) + (4.0 / 29.0))

    fx, fy, fz = f(xyz_ref[0]), f(xyz_ref[1]), f(xyz_ref[2])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.array([L, a, b])

def lab_to_xyz(lab, xyz_n):
    """Convertit les coordonnées CIE L*a*b* en CIE XYZ."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    def f_inv(t):
        delta = 6.0 / 29.0
        return np.where(t > delta,
                        t**3,
                        3.0 * delta**2 * (t - 4.0 / 29.0))

    xyz_ref = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1)
    return xyz_ref * xyz_n

def xyz_to_srgb(xyz):
    """Convertit CIE XYZ en sRGB."""
    xyz_norm = xyz / 100.0
    rgb_linear = np.dot(xyz_norm, XYZ_TO_RGB_MATRIX.T)
    
    # Clip les valeurs linéaires avant la correction gamma pour éviter les erreurs sur les nombres négatifs
    rgb_linear = np.clip(rgb_linear, 0, 1)
    
    def gamma_correct(c):
        return np.where(c <= 0.0031308,
                        12.92 * c,
                        1.055 * (c**(1.0/2.4)) - 0.055)
        
    rgb = gamma_correct(rgb_linear)
    return np.clip(rgb, 0, 1)

def plot_cielab_diagram(ax, lab_nominal, lab_mc_points):
    """Génère le diagramme de chromaticité a*b* et y superpose les points."""
    a_range = (-100, 100)
    b_range = (-100, 100)
    
    # Créer une image de fond
    a_grid, b_grid = np.meshgrid(np.linspace(a_range[0], a_range[1], 200),
                                 np.linspace(b_range[0], b_range[1], 200))
    
    L_grid = np.full(a_grid.shape, 75.0) # Luminance constante pour la visualisation
    lab_image = np.stack([L_grid, a_grid, b_grid], axis=-1)
    
    xyz_image = lab_to_xyz(lab_image, XYZ_N_D65)
    srgb_image = xyz_to_srgb(xyz_image)
    
    ax.imshow(srgb_image, origin='lower', extent=[*a_range, *b_range], aspect='equal')
    
    # Superposer les points
    ax.scatter(lab_mc_points[:, 1], lab_mc_points[:, 2], alpha=0.4, s=25, c='white', edgecolor='black', linewidth=0.5, label=f'Tirages Monte-Carlo (σ=2nm)')
    ax.scatter(lab_nominal[1], lab_nominal[2], color='red', marker='*', s=250, zorder=5, edgecolor='black', label=f'Point Nominal')
    
    ax.axhline(0, color='grey', linewidth=0.5); ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_title("Espace CIELAB (a*, b*) pour la Réflectance")
    ax.set_xlabel("a* (rouge ↔ vert)"); ax.set_ylabel("b* (jaune ↔ bleu)")
    ax.grid(True, linestyle=':'); ax.set_aspect('equal', adjustable='box'); ax.legend()
    ax.set_xlim(a_range); ax.set_ylim(b_range)

# --- SECTION : MOTEUR DE CALCUL PHYSIQUE (JAX) ---

MaterialInputType = Union[dict, complex, float, int]

def get_cauchy_coefficients(n_400, n_700):
    """Calcule les coefficients A et B du modèle de Cauchy n(lambda) = A + B/lambda^2."""
    lambda1_sq_inv = 1.0 / (400.0**2)
    lambda2_sq_inv = 1.0 / (700.0**2)

    if abs(lambda1_sq_inv - lambda2_sq_inv) < 1e-9: # Précision simple
        return n_400, 0.0

    B = (n_400 - n_700) / (lambda1_sq_inv - lambda2_sq_inv)
    A = n_400 - B * lambda1_sq_inv
    return A, B

def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                 l_vec_target_jnp: jnp.ndarray) -> Tuple[Optional[jnp.ndarray], List[str]]:
    """Calcule le tableau d'indices de réfraction pour un vecteur de longueurs d'onde."""
    logs = []
    try:
        if isinstance(material_definition, dict) and material_definition.get('type') == 'cauchy':
            n_400 = material_definition['n_400']
            n_700 = material_definition['n_700']
            A, B = get_cauchy_coefficients(n_400, n_700)

            n_real_array = A + B / (l_vec_target_jnp**2)
            result = n_real_array.astype(DTYPE_COMPLEX)

        elif isinstance(material_definition, (complex, float, int)):
            nk_complex = jnp.asarray(material_definition, dtype=DTYPE_COMPLEX)
            if nk_complex.real <= 0:
                logs.append(f"AVERTISSEMENT : Indice constant ({nk_complex.real}) <= 0. Utilisation de 1.0.")
                nk_complex = complex(1.0, 0.0)
            nk_complex = complex(nk_complex.real, 0.0)
            result = jnp.full(l_vec_target_jnp.shape, nk_complex, dtype=DTYPE_COMPLEX)
        else:
            raise TypeError(f"Type de définition de matériau non supporté : {type(material_definition)}")

        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"AVERTISSEMENT : Indice <=0 ou NaN détecté. Remplacé par 1.0.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 0.0j, result)

        return result.astype(DTYPE_COMPLEX), logs
    except Exception as e:
        logs.append(f"Erreur lors de la préparation des données du matériau pour '{material_definition}': {e}")
        st.error(f"Erreur critique lors de la préparation du matériau '{material_definition}': {e}")
        return None, logs

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float) -> Tuple[Optional[complex], List[str]]:
    """Obtient l'indice de réfraction complexe pour une seule longueur d'onde."""
    logs = []
    if l_nm_target <= 0:
        logs.append(f"Erreur : Longueur d'onde cible {l_nm_target}nm invalide pour obtenir n+ik.")
        return None, logs
    l_vec_jnp = jnp.array([l_nm_target], dtype=DTYPE_FLOAT)
    nk_array, prep_logs = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp)
    logs.extend(prep_logs)
    if nk_array is None:
        return None, logs
    else:
        nk_complex = complex(nk_array[0])
        return nk_complex, logs

@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Étape de calcul de la matrice de transfert pour une couche (utilisé dans `scan`)."""
    thickness, Ni, l_val = layer_data
    eta = Ni
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta) < 1e-9, 1e-9 + 0j, eta)
        m01 = (1j / safe_eta) * sin_phi
        m10 = 1j * eta * sin_phi
        M_layer = jnp.array([[cos_phi, m01], [m10, cos_phi]], dtype=DTYPE_COMPLEX)
        return M_layer @ carry_matrix
    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix
    new_matrix = cond(thickness > 1e-9, compute_M_layer, compute_identity, thickness)
    return new_matrix, None

@jax.jit
def compute_stack_matrix_core_jax(ep_vector: jnp.ndarray, layer_indices: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice de transfert totale pour un empilement."""
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=DTYPE_COMPLEX)
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_TR_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                        layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calcule la transmittance (T) et la réflectance (R) pour une seule longueur d'onde."""
    etainc = jnp.array(1.0 + 0j, dtype=DTYPE_COMPLEX)
    etasub = nSub_at_lval

    def calculate_for_valid_l(l_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        safe_denominator = jnp.where(jnp.abs(denominator) < 1e-9, 1e-9 + 0j, denominator)

        # Transmittance
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)

        # Réflectance
        numerator_rs = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
        rs = numerator_rs / safe_denominator
        Rs = rs * jnp.conj(rs)
        Rs = jnp.real(Rs)

        return jnp.where(jnp.abs(denominator) < 1e-9, jnp.nan, Ts), jnp.where(jnp.abs(denominator) < 1e-9, jnp.nan, Rs)

    def calculate_for_invalid_l(l_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.nan, jnp.nan

    return cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)

def calculate_TR_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                             stack: List[Dict],
                             materials: Dict[str, Dict],
                             nSub_material: MaterialInputType,
                             l_vec: Union[np.ndarray, List[float]],
                             backside_enabled: bool = False,
                             backside_stack: Optional[List[Dict]] = None,
                             backside_l0: Optional[float] = None
                             ) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """Calcule T et R pour un vecteur de longueurs d'onde, en tenant compte de la face arrière si activée."""
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=DTYPE_FLOAT)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=DTYPE_FLOAT)
    if not l_vec_jnp.size:
        return {'l': np.array([]), 'Ts': np.array([]), 'Rs': np.array([])}, ["Vecteur lambda vide."]

    # --- Calcul pour la structure vide (substrat nu) ---
    if not stack or not ep_vector_jnp.size:
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp)
        logs.extend(logs_sub)
        if nSub_arr is None: return None, logs

        r = (1.0 - nSub_arr) / (1.0 + nSub_arr)
        Rs = (r * jnp.conj(r)).real
        Ts = 1.0 - Rs
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts), 'Rs': np.array(Rs)}, logs

    # --- Préparation des indices de réfraction pour toutes les couches ---
    material_arrays = {}
    for key, props in materials.items():
        arr, log = _get_nk_array_for_lambda_vec(props, l_vec_jnp)
        logs.extend(log)
        if arr is None:
            logs.append(f"Erreur critique : Échec du chargement des indices du matériau {key}.")
            return None, logs
        material_arrays[key] = arr

    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp)
    logs.extend(logs_sub)
    if nSub_arr is None:
        logs.append("Erreur critique : Échec du chargement des indices du substrat.")
        return None, logs

    # --- Calcul de la face avant (Ta, Ra) ---
    front_indices_list = [material_arrays[layer['material']] for layer in stack]
    front_layer_indices_array = jnp.stack(front_indices_list, axis=0)
    front_layer_indices_array_T = front_layer_indices_array.T

    Ta_arr_raw, Ra_arr_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, front_layer_indices_array_T, nSub_arr
    )
    Ta_arr = jnp.clip(jnp.nan_to_num(Ta_arr_raw, nan=0.0), 0.0, 1.0)
    Ra_arr = jnp.clip(jnp.nan_to_num(Ra_arr_raw, nan=0.0), 0.0, 1.0)

    # --- Calcul optionnel de la face arrière et combinaison ---
    if backside_enabled:
        logs.append("Calcul de la face arrière activé.")

        if backside_stack and backside_l0 is not None:
            # Calcul pour un empilement sur la face arrière
            backside_ep, logs_ep_back = calculate_initial_ep(backside_stack, backside_l0, materials)
            logs.extend(logs_ep_back)
            if backside_ep is None:
                logs.append("Erreur critique : Échec du calcul des épaisseurs de la face arrière.")
                return None, logs

            backside_ep_jnp = jnp.asarray(backside_ep, dtype=DTYPE_FLOAT)

            back_indices_list = [material_arrays[layer['material']] for layer in backside_stack]
            back_layer_indices_array = jnp.stack(back_indices_list, axis=0)
            back_layer_indices_array_T = back_layer_indices_array.T

            Tb_arr_raw, Rb_arr_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                l_vec_jnp, backside_ep_jnp, back_layer_indices_array_T, nSub_arr
            )
            Tb_arr = jnp.clip(jnp.nan_to_num(Tb_arr_raw, nan=0.0), 0.0, 1.0)
            Rb_arr = jnp.clip(jnp.nan_to_num(Rb_arr_raw, nan=0.0), 0.0, 1.0)
        else:
            # Calcul pour un substrat nu sur la face arrière
            logs.append("Aucun empilement sur la face arrière, calcul de la réflexion du substrat nu.")
            r_back = (nSub_arr - 1.0) / (nSub_arr + 1.0)
            Rb_arr = (r_back * jnp.conj(r_back)).real
            Tb_arr = 1.0 - Rb_arr # En supposant un substrat non absorbant

        # Combinaison des deux faces
        denominator = 1.0 - Ra_arr * Rb_arr
        safe_denominator = jnp.where(denominator < 1e-9, 1e-9, denominator)

        Ts_total = (Ta_arr * Tb_arr) / safe_denominator
        Rs_total = Ra_arr + (Ta_arr**2 * Rb_arr) / safe_denominator

        Ts_arr = jnp.clip(Ts_total, 0.0, 1.0)
        Rs_arr = jnp.clip(Rs_total, 0.0, 1.0)

        logs.append("Spectres des faces avant et arrière combinés.")
    else:
        Ts_arr = Ta_arr
        Rs_arr = Ra_arr

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr), 'Rs': np.array(Rs_arr)}, logs


def calculate_initial_ep(stack: List[Dict], l0: float,
                         materials: Dict[str, Dict]) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calcule les épaisseurs physiques initiales à partir des épaisseurs QWOT."""
    logs = []
    num_layers = len(stack)
    ep_initial = np.zeros(num_layers, dtype=np.float32)
    if l0 <= 0:
        logs.append(f"AVERTISSEMENT : l0={l0} <= 0 dans calculate_initial_ep. Épaisseurs initiales mises à 0.")
        return ep_initial, logs

    for i, layer in enumerate(stack):
        material_key = layer['material']
        material_props = materials.get(material_key)
        if not material_props:
            logs.append(f"Erreur: Matériau '{material_key}' non trouvé pour la couche {i+1}.")
            return None, logs
            
        n_complex_at_l0, log = _get_nk_at_lambda(material_props, l0)
        logs.extend(log)

        if n_complex_at_l0 is None:
            logs.append(f"Erreur : Impossible d'obtenir l'indice pour le matériau {material_key} à l0={l0}nm.")
            st.error(f"Erreur critique lors de l'obtention de l'indice pour {material_key} à l0={l0}nm.")
            return None, logs

        n_real_at_l0 = n_complex_at_l0.real
        if n_real_at_l0 <= 1e-9:
            logs.append(f"AVERTISSEMENT : Indice pour {material_key} à l0={l0}nm est <= 0. Le calcul de l'épaisseur peut être incorrect.")
            ep_initial[i] = 0.0
        else:
            ep_initial[i] = layer['qwot'] * l0 / (4.0 * n_real_at_l0)

    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-9) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        logs.append(f"AVERTISSEMENT : {num_clamped_zero} épaisseurs initiales < {MIN_THICKNESS_PHYS_NM}nm ont été mises à 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)

    return ep_initial.astype(np.float32), logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, stack: List[Dict], l0: float,
                           materials: Dict[str, Dict]) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calcule les épaisseurs QWOT à partir des épaisseurs physiques."""
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float32)
    if l0 <= 0:
        logs.append(f"AVERTISSEMENT : l0={l0} <= 0 dans calculate_qwot_from_ep. QWOT mis à NaN.")
        return qwot_multipliers, logs

    for i, layer in enumerate(stack):
        material_key = layer['material']
        material_props = materials[material_key]
        n_complex_at_l0, log = _get_nk_at_lambda(material_props, l0)
        logs.extend(log)

        if n_complex_at_l0 is None:
            logs.append(f"Erreur : Impossible d'obtenir l'indice pour {material_key} à l0={l0}nm pour calculer le QWOT.")
            st.error(f"Erreur lors du calcul du QWOT pour {material_key}.")
            continue # Laisser en NaN

        n_real_at_l0 = n_complex_at_l0.real
        if n_real_at_l0 <= 1e-9:
            if ep_vector[i] > 1e-9 :
                logs.append(f"AVERTISSEMENT : Impossible de calculer le QWOT pour la couche {i+1} ({material_key}) car l'indice({l0}nm) <= 0.")
            else:
                qwot_multipliers[i] = 0.0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_at_l0) / l0

    if np.any(np.isnan(qwot_multipliers)):
        st.warning("Certaines valeurs QWOT n'ont pas pu être calculées (indices invalides à l0). Elles apparaissent comme NaN.")

    return qwot_multipliers.astype(np.float32), logs

def calculate_final_rmse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    """Calcule le RMSE pondéré entre les résultats calculés et les cibles."""
    total_weighted_squared_error = 0.0
    total_weight = 0.0
    total_points_in_targets = 0
    rmse = None
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return rmse, total_points_in_targets
    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])
    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
        return rmse, total_points_in_targets
    for target in active_targets:
        try:
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            weight = float(target.get('weight', 1.0))
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue
            if l_max < l_min: continue
        except (KeyError, ValueError, TypeError):
            continue
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]
            if calculated_Ts_in_zone.size == 0: continue
            if abs(l_max - l_min) < 1e-9:
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_weighted_squared_error += np.sum(squared_errors * weight)
            current_points_count = len(calculated_Ts_in_zone)
            total_weight += current_points_count * weight
            total_points_in_targets += current_points_count

    if total_weight > 0:
        mse = total_weighted_squared_error / total_weight
        rmse = np.sqrt(mse)
    return rmse, total_points_in_targets

@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 layer_indices_array: jnp.ndarray,
                                                 nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float, float], ...],
                                                 min_thickness_phys_nm: float,
                                                 backside_enabled: bool,
                                                 backside_ep: jnp.ndarray,
                                                 backside_layer_indices: jnp.ndarray
                                                 ) -> jnp.ndarray:
    """Fonction de coût (MSE pondéré + pénalité) pour l'optimisation, compilée avec JAX."""
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-9)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_weight = 1e5
    penalty_cost = penalty_thin * penalty_weight
    ep_vector_calc = jnp.maximum(ep_vector, 0.0)

    layer_indices_array_T = layer_indices_array.T

    # Calcul pour la face avant
    Ta_raw, Ra_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, layer_indices_array_T, nSub_arr
    )
    Ta = jnp.clip(jnp.nan_to_num(Ta_raw, nan=0.0), 0.0, 1.0)
    Ra = jnp.clip(jnp.nan_to_num(Ra_raw, nan=0.0), 0.0, 1.0)

    # Calcul conditionnel pour la face arrière
    def with_backside(Ta, Ra):
        backside_layer_indices_T = backside_layer_indices.T
        Tb_raw, Rb_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
            l_vec_optim, backside_ep, backside_layer_indices_T, nSub_arr
        )
        Tb = jnp.clip(jnp.nan_to_num(Tb_raw, nan=0.0), 0.0, 1.0)
        Rb = jnp.clip(jnp.nan_to_num(Rb_raw, nan=0.0), 0.0, 1.0)

        denominator = 1.0 - Ra * Rb
        safe_denominator = jnp.where(denominator < 1e-9, 1e-9, denominator)
        T_total = (Ta * Tb) / safe_denominator
        return jnp.clip(T_total, 0.0, 1.0)

    def without_backside(Ta, Ra):
        return Ta

    Ts = cond(backside_enabled, with_backside, without_backside, Ta, Ra)

    total_weighted_squared_error = 0.0
    total_weight = 0.0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max, weight = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)

        total_weighted_squared_error += jnp.sum(masked_sq_error * weight)
        total_weight += jnp.sum(target_mask) * weight

    mse = jnp.where(total_weight > 1e-9, # Utiliser une petite tolérance pour la robustesse
                                          total_weighted_squared_error / total_weight,
                                          jnp.inf)
    final_cost = mse + penalty_cost
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)

# --- SECTION : LOGIQUE D'OPTIMISATION ET DE MANIPULATION DE STRUCTURE ---

def log_optimization_setup(stack, ep_start, variable_indices, fixed_indices):
    """Génère des logs détaillés sur la configuration de l'optimisation."""
    num_layers = len(stack)
    ep_start_variable = ep_start[variable_indices] if len(variable_indices) > 0 else np.array([])
    ep_start_fixed = ep_start[fixed_indices] if len(fixed_indices) > 0 else np.array([])
    
    add_log("--- CONFIGURATION DE L'OPTIMISATION ---")
    add_log(f"Nombre total de couches: {num_layers}")
    add_log(f"Indices des couches variables: {[i + 1 for i in variable_indices]}")
    add_log(f"Indices des couches fixes: {[i + 1 for i in fixed_indices]}")
    add_log(f"!!! L'OPTIMISEUR TRAVAILLE SUR {len(variable_indices)} VARIABLES. !!!")
    add_log(f"Épaisseurs initiales (toutes): {np.round(ep_start, 2).tolist()}")
    add_log(f"Épaisseurs initiales (variables): {np.round(ep_start_variable, 2).tolist()}")
    add_log(f"Épaisseurs initiales (fixes): {np.round(ep_start_fixed, 2).tolist()}")
    add_log("---------------------------------------")

def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str]:
    """Exécute l'optimisation locale (L-BFGS-B) sur les couches variables."""
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimisation non lancée ou échouée prématurément."
    final_ep = None
    if num_layers_start == 0:
        logs.append(f"{log_prefix}Impossible d'optimiser une structure vide.")
        return None, False, np.inf, logs, "Structure vide"
    try:
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        materials = validated_inputs['materials']
        nSub_material = validated_inputs['nSub_material']
        stack = validated_inputs['stack']

        variable_indices = [i for i, layer in enumerate(stack) if layer['is_variable']]
        fixed_indices = [i for i, layer in enumerate(stack) if not layer['is_variable']]

        log_optimization_setup(stack, ep_start_optim, variable_indices, fixed_indices)

        if not variable_indices:
            logs.append(f"{log_prefix}Aucune couche variable à optimiser.")
            return ep_start_optim, True, 0.0, logs, "Aucune couche variable."

        ep_start_variable = ep_start_optim[variable_indices]
        ep_start_fixed = ep_start_optim[fixed_indices]

        num_pts_optim = validated_inputs['optim_resolution_points']
        l_vec_optim_np_base = np.linspace(l_min_optim, l_max_optim, num_pts_optim)
        
        target_points = []
        for t in active_targets:
            target_points.extend([t['min'], t['max']])
        l_vec_optim_np = np.unique(np.concatenate([l_vec_optim_np_base, np.array(target_points)]))

        l_vec_optim_jax = jnp.asarray(l_vec_optim_np, dtype=DTYPE_FLOAT)

        material_arrays = {key: _get_nk_array_for_lambda_vec(props, l_vec_optim_jax)[0] for key, props in materials.items()}
        nSub_arr_optim, _ = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        indices_list = [material_arrays[layer['material']] for layer in stack]
        layer_indices_array = jnp.stack(indices_list, axis=0)

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max']), float(t.get('weight', 1.0))) for t in active_targets)

        backside_enabled = validated_inputs.get('backside_enabled', False)
        backside_ep_jnp = jnp.array([], dtype=DTYPE_FLOAT)
        num_wavelengths = len(l_vec_optim_jax)
        backside_layer_indices_jnp = jnp.empty((0, num_wavelengths), dtype=DTYPE_COMPLEX)

        if backside_enabled:
            backside_stack = validated_inputs['backside_stack']
            if backside_stack:
                backside_ep, _ = calculate_initial_ep(backside_stack, validated_inputs['backside_l0'], materials)
                backside_ep_jnp = jnp.asarray(backside_ep, dtype=DTYPE_FLOAT)
                back_indices_list = [material_arrays[layer['material']] for layer in backside_stack]
                backside_layer_indices_jnp = jnp.stack(back_indices_list, axis=0)

        cost_fn_with_grad = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        def scipy_obj_grad_wrapper(ep_variable_np: np.ndarray, *args: Any) -> Tuple[float, np.ndarray]:
            try:
                (
                    full_ep_template_np,
                    variable_indices_np,
                    layer_indices_array_jax, nSub_arr_jax, l_vec_jax,
                    active_targets_tup, min_thick_phys, backside_en,
                    backside_ep_jax, backside_indices_jax
                ) = args

                ep_full_jnp = jnp.array(full_ep_template_np)
                ep_full_jnp = ep_full_jnp.at[variable_indices_np].set(jnp.asarray(ep_variable_np))
                
                value_jax, grad_full_jax = cost_fn_with_grad(
                    ep_full_jnp, layer_indices_array_jax, nSub_arr_jax, l_vec_jax,
                    active_targets_tup, min_thick_phys, backside_en,
                    backside_ep_jax, backside_indices_jax
                )

                grad_variable_jax = grad_full_jax.take(variable_indices_np)

                value_np = np.array(value_jax, dtype=np.float64).item()
                grad_variable_np = np.array(grad_variable_jax, dtype=np.float64)

                if not np.isfinite(value_np):
                    return np.inf, np.zeros_like(grad_variable_np)

                return value_np, grad_variable_np
            except Exception as e_wrap:
                print(f"CRITICAL ERROR in scipy_obj_grad_wrapper: {e_wrap}")
                print(f"Traceback: {traceback.format_exc(limit=4)}")
                print(f"ep_variable causing error: {ep_variable_np}")
                return np.inf, np.zeros_like(ep_variable_np, dtype=np.float64)

        full_ep_template = np.zeros(num_layers_start, dtype=np.float32)
        if fixed_indices:
            full_ep_template[fixed_indices] = ep_start_fixed
        
        static_args_for_scipy = (
            full_ep_template,
            np.array(variable_indices),
            layer_indices_array, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys, backside_enabled,
            backside_ep_jnp, backside_layer_indices_jnp
        )

        lbfgsb_bounds = [(0.0, None)] * len(variable_indices)
        options = {'maxiter': MAXITER_HARDCODED, 'maxfun': MAXFUN_HARDCODED, 'disp': False, 'ftol': 1e-9, 'gtol': 1e-7}

        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_variable,
                          args=static_args_for_scipy,
                          method='L-BFGS-B',
                          jac=True,
                          bounds=lbfgsb_bounds,
                          options=options)

        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)

        final_ep_full = np.copy(ep_start_optim)
        if result.success or result.status == 1:
            final_ep_full[variable_indices] = result.x
            final_ep = np.where(final_ep_full < min_thickness_phys, 0.0, final_ep_full)
            optim_success = True
            log_status = "succès" if result.success else "limite atteinte"
            logs.append(f"{log_prefix}Optimisation terminée ({log_status}). Coût final (MSE) : {final_cost:.3e}, Msg : {result_message_str}")
        else:
            final_ep = np.where(ep_start_optim < min_thickness_phys, 0.0, ep_start_optim)
            optim_success = False
            logs.append(f"{log_prefix}Optimisation ÉCHOUÉE. Statut : {result.status}, Msg : {result_message_str}, Coût : {final_cost:.3e}")

    except Exception as e_optim:
        logs.append(f"{log_prefix}ERREUR MAJEURE pendant l'optimisation JAX/Scipy : {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Erreur critique pendant l'optimisation : {e_optim}")
        final_ep = np.where(ep_start_optim < min_thickness_phys, 0.0, ep_start_optim) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception : {e_optim}"
    return final_ep, optim_success, final_cost, logs, result_message_str

def _perform_layer_removal(ep_vector_in: np.ndarray, stack_in: List[Dict], min_thickness_phys: float,
                           log_prefix: str = "") -> Tuple[Optional[np.ndarray], Optional[List[Dict]], bool, List[str]]:
    """Identifie la couche la plus fine, la supprime et fusionne les couches adjacentes si leurs matériaux sont identiques."""
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)

    if num_layers == 0:
        return current_ep, stack_in, False, [f"{log_prefix}Structure vide, impossible de supprimer."]

    candidate_indices = np.where(current_ep > 1e-9)[0] # On considère toutes les couches non nulles
    if candidate_indices.size == 0:
        logs.append(f"{log_prefix}Aucune couche non-nulle trouvée à supprimer.")
        return current_ep, stack_in, False, logs

    min_idx_local = np.argmin(current_ep[candidate_indices])
    thin_layer_index = candidate_indices[min_idx_local]

    logs.append(f"{log_prefix}Couche la plus fine identifiée pour action : Index {thin_layer_index} (Couche {thin_layer_index + 1}), épaisseur {current_ep[thin_layer_index]:.3f} nm.")

    # Cas 1 : Couche interne, vérifier la fusion
    if 0 < thin_layer_index < num_layers - 1:
        mat_before = stack_in[thin_layer_index - 1]['material']
        mat_after = stack_in[thin_layer_index + 1]['material']
        if mat_before == mat_after:
            logs.append(f"{log_prefix}Matériaux adjacents ('{mat_before}') identiques. Fusion des couches {thin_layer_index} et {thin_layer_index + 2}.")

            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]

            # Mettre à jour le vecteur ep
            ep_temp = np.delete(current_ep, [thin_layer_index - 1, thin_layer_index, thin_layer_index + 1])
            new_ep = np.insert(ep_temp, thin_layer_index - 1, merged_thickness)

            # Mettre à jour l'empilement
            stack_before = stack_in[:thin_layer_index - 1]
            stack_after = stack_in[thin_layer_index + 2:]
            merged_layer = stack_in[thin_layer_index - 1].copy() # Garder les propriétés de la première couche
            merged_layer['qwot'] = -1 # Le QWOT est maintenant invalide, nécessite un recalcul ultérieur
            new_stack = stack_before + [merged_layer] + stack_after

            return new_ep, new_stack, True, logs

    # Cas 2 : Pas de fusion (couche de bord ou matériaux adjacents différents)
    logs.append(f"{log_prefix}Suppression simple de la couche {thin_layer_index + 1}.")
    new_ep = np.delete(current_ep, thin_layer_index)
    new_stack = [layer for i, layer in enumerate(stack_in) if i != thin_layer_index]

    return new_ep, new_stack, True, logs


def validate_targets() -> Optional[List[Dict]]:
    """Valide les cibles spectrales saisies par l'utilisateur."""
    active_targets = []
    logs = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Erreur interne : Liste de cibles manquante ou invalide dans session_state.")
        return None
    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False):
            try:
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])
                weight = float(target_state.get('weight', 1.0))
                if l_max < l_min:
                    logs.append(f"Cible {i+1} Erreur : λ max ({l_max:.1f}) < λ min ({l_min:.1f}).")
                    is_valid = False; continue
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                    logs.append(f"Cible {i+1} Erreur : Transmittance hors de [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}).")
                    is_valid = False; continue
                if weight < 0:
                    logs.append(f"Cible {i+1} Erreur : Le poids doit être non-négatif (Poids={weight:.2f}).")
                    is_valid = False; continue
                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max,
                    'weight': weight
                })
            except (KeyError, ValueError, TypeError) as e:
                logs.append(f"Cible {i+1} Erreur : Données manquantes ou invalides ({e}).")
                is_valid = False; continue
    if not is_valid:
        st.warning("Des erreurs existent dans les définitions des cibles spectrales actives. Veuillez corriger.")
        return None
    elif not active_targets:
        return []
    else:
        return active_targets

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    """Détermine la plage de longueurs d'onde globale à partir des cibles actives."""
    overall_min, overall_max = None, None
    if validated_targets:
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max

def calculate_optim_points(active_targets: List[Dict]) -> int:
    """Calcule le nombre de points pour l'optimisation basé sur un intervalle de ~20nm."""
    if not active_targets:
        return 50 # Valeur par défaut si aucune cible n'est active
    
    l_min, l_max = get_lambda_range_from_targets(active_targets)
    
    if l_min is None or l_max is None or l_max <= l_min:
        return 50 # Valeur par défaut pour des cas invalides

    spectral_range = l_max - l_min
    # Calcule le nombre de points pour avoir un intervalle d'environ 20nm
    num_points = int(np.ceil(spectral_range / 20.0))
    
    # Assure un minimum de points pour la robustesse
    return max(25, num_points)

# --- SECTION : GESTION DE L'ÉTAT DE L'APPLICATION ---

def initialize_session_state():
    """Initialise toutes les variables de session si elles n'existent pas."""
    if 'init_done' not in st.session_state:
        st.session_state.log_messages = deque(maxlen=200)
        st.session_state.current_ep = None
        st.session_state.stack = [
            {'material': 'H', 'qwot': 1.0, 'is_variable': True},
            {'material': 'L', 'qwot': 1.0, 'is_variable': True},
            {'material': 'H', 'qwot': 1.0, 'is_variable': True},
            {'material': 'L', 'qwot': 1.0, 'is_variable': True}
        ]
        st.session_state.backside_stack = []
        st.session_state.optimized_ep = None
        st.session_state.is_optimized_state = False
        st.session_state.optimized_qwot_str = ""
        st.session_state.optimized_stack_sequence = None
        st.session_state.ep_history = deque(maxlen=5)
        st.session_state.last_rmse = None
        st.session_state.needs_rerun_calc = False
        st.session_state.rerun_calc_params = {}
        st.session_state.action = None
        st.session_state.monte_carlo_results = None
        st.session_state.tolerance_analysis_results = None
        st.session_state.color_results = None

        st.session_state.l0 = 500.0
        st.session_state.backside_l0 = 500.0
        st.session_state.auto_thin_threshold = 1.0
        st.session_state.auto_scale_y = True
        st.session_state.monte_carlo_std_dev = 2.0
        st.session_state.backside_enabled = False
        st.session_state.de_action_radius = 1.0
        
        st.session_state.de_speed = "Normal"

        st.session_state.targets = [
            {'enabled': True, 'min': 380.0, 'max': 760.0, 'target_min': 1.0, 'target_max': 1.0, 'weight': 1.0},
            {'enabled': False, 'min': 400.0, 'max': 500.0, 'target_min': 0.0, 'target_max': 0.0, 'weight': 1.0},
            {'enabled': False, 'min': 600.0, 'max': 700.0, 'target_min': 0.0, 'target_max': 0.0, 'weight': 1.0},
        ]

        st.session_state.materials = {
            'H': {'type': 'cauchy', 'n_400': 2.35, 'n_700': 2.35},
            'L': {'type': 'cauchy', 'n_400': 1.46, 'n_700': 1.46},
            'A': {'type': 'cauchy', 'n_400': 2.05, 'n_700': 2.05},
            'B': {'type': 'cauchy', 'n_400': 1.75, 'n_700': 1.75},
            'C': {'type': 'cauchy', 'n_400': 1.60, 'n_700': 1.60},
            'Substrate': {'type': 'cauchy', 'n_400': 1.52, 'n_700': 1.52}
        }

        st.session_state.init_done = True
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Chargement initial"}
        add_log("Application initialisée.")

def clear_optimized_state():
    """Réinitialise l'état optimisé."""
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.optimized_qwot_str = ""
    st.session_state.optimized_stack_sequence = None
    st.session_state.last_rmse = None
    st.session_state.monte_carlo_results = None
    st.session_state.color_results = None

def trigger_nominal_recalc():
    """Déclenche un recalcul de la structure nominale après une modification de l'interface."""
    if not st.session_state.get('calculating', False):
        clear_optimized_state()
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False,
            'method_name': "Structure Avant (Mise à jour Param.)",
            'force_ep': None
        }

# --- SECTION : ACTIONS ET LOGIQUE PRINCIPALE ---

def update_nominal_stack_from_ep(ep_vector: np.ndarray, stack_sequence: List[Dict], l0: float, materials: Dict) -> bool:
    """Met à jour st.session_state.stack à partir d'un vecteur d'épaisseurs physiques."""
    add_log("--- Mise à jour auto. de la structure nominale depuis les épaisseurs optimisées ---")
    new_qwots, logs_qwot = calculate_qwot_from_ep(ep_vector, stack_sequence, l0, materials)
    add_log(logs_qwot)

    if new_qwots is None or np.any(np.isnan(new_qwots)):
        st.error("Impossible de calculer les nouvelles valeurs QWOT. La structure nominale n'a pas été mise à jour.")
        add_log("Erreur : Échec de la conversion des épaisseurs optimisées en QWOT.")
        return False

    new_stack = []
    for i, layer in enumerate(stack_sequence):
        new_layer = layer.copy()
        if i < len(new_qwots):
            new_layer['qwot'] = new_qwots[i]
            new_stack.append(new_layer)
        else:
            add_log(f"AVERTISSEMENT: Incohérence de longueur entre stack ({len(stack_sequence)}) et new_qwots ({len(new_qwots)}). Couche {i+1} ignorée.")

    st.session_state.stack = new_stack
    add_log(f"Nouvelle structure avant définie avec {len(new_stack)} couches.")
    clear_optimized_state()
    return True

def run_calculation_wrapper(is_optimized_run: bool, method_name: str, force_ep: Optional[np.ndarray] = None):
    """
    Wrapper pour effectuer un calcul complet, mettant à jour l'état de la session avec les résultats.
    """
    add_log(f"--- Calcul déclenché par : {method_name} ---")
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    st.session_state.monte_carlo_results = None
    st.session_state.color_results = None # Réinitialiser les résultats couleur aussi

    with st.spinner("Calcul du spectre en cours..."):
        try:
            active_targets = validate_targets()
            if active_targets is None:
                st.error("Calcul annulé : Cibles invalides.")
                add_log("Calcul annulé : Cibles invalides.")
                return

            l_min_calc, l_max_calc = get_lambda_range_from_targets(active_targets)
            if not active_targets:
                l_min_calc, l_max_calc = 380.0, 760.0
            if l_min_calc is None or l_max_calc is None:
                st.error("Impossible de déterminer la plage de longueurs d'onde pour le calcul.")
                add_log("Impossible de déterminer la plage de longueurs d'onde pour le calcul.")
                return

            materials_calc = st.session_state.materials.copy()
            nSub_material_calc = materials_calc.pop('Substrate')
            stack_calc = st.session_state.stack
            l0_calc = st.session_state.l0

            ep_to_use = None
            if force_ep is not None:
                ep_to_use = force_ep
                add_log(f"Utilisation du vecteur d'épaisseurs forcé ({len(ep_to_use)} couches).")
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                ep_to_use = st.session_state.optimized_ep
                stack_calc = st.session_state.get('optimized_stack_sequence', st.session_state.stack)
                add_log(f"Utilisation du vecteur d'épaisseurs optimisé ({len(ep_to_use)} couches).")
            else:
                if not stack_calc:
                    ep_to_use = np.array([])
                else:
                    ep_to_use, logs_ep = calculate_initial_ep(stack_calc, l0_calc, materials_calc)
                    add_log(logs_ep)
                if ep_to_use is None:
                    st.error("Échec du calcul des épaisseurs physiques initiales pour la structure avant.")
                    add_log("Échec du calcul des épaisseurs physiques initiales.")
                    return
                add_log(f"Utilisation du vecteur d'épaisseurs de la structure avant ({len(ep_to_use)} couches).")

            st.session_state.current_ep = ep_to_use.copy()

            num_pts_optim_base = calculate_optim_points(active_targets)
            
            # --- FINE GRID CALCULATION (pour la courbe bleue lisse) ---
            num_pts_fine = num_pts_optim_base * 10
            add_log(f"Calcul du spectre d'affichage avec {num_pts_fine} points.")
            l_vec_fine = np.linspace(l_min_calc, l_max_calc, num_pts_fine)

            res_fine, logs_calc = calculate_TR_from_ep_jax(
                ep_to_use,
                stack_calc,
                materials_calc,
                nSub_material_calc,
                l_vec_fine,
                backside_enabled=st.session_state.get('backside_enabled', False),
                backside_stack=st.session_state.backside_stack,
                backside_l0=st.session_state.backside_l0
            )
            add_log(logs_calc)

            if res_fine is None:
                st.error("Le calcul principal a échoué. Vérifiez les logs pour plus de détails.")
                add_log("Le calcul principal a échoué.")
                return

            # --- OPTIMIZATION GRID CALCULATION (pour les croix rouges) ---
            res_optim_grid = None
            if active_targets:
                l_vec_optim_np_base = np.linspace(l_min_calc, l_max_calc, num_pts_optim_base)
                target_points = []
                for t in active_targets:
                    target_points.extend([t['min'], t['max']])
                l_vec_optim_np = np.unique(np.concatenate([l_vec_optim_np_base, np.array(target_points)]))
                add_log(f"Calcul du spectre sur la grille d'optimisation ({len(l_vec_optim_np)} points).")

                res_optim_grid, logs_calc_optim = calculate_TR_from_ep_jax(
                    ep_to_use,
                    stack_calc,
                    materials_calc,
                    nSub_material_calc,
                    l_vec_optim_np,
                    backside_enabled=st.session_state.get('backside_enabled', False),
                    backside_stack=st.session_state.backside_stack,
                    backside_l0=st.session_state.backside_l0
                )
                add_log(logs_calc_optim)
                if res_optim_grid is None:
                    add_log("AVERTISSEMENT: Le calcul sur la grille d'optimisation a échoué.")

            final_rmse, points_in_targets = calculate_final_rmse(res_fine, active_targets)
            st.session_state.last_rmse = final_rmse
            add_log(f"Calcul terminé. RMSE final : {final_rmse:.4e} sur {points_in_targets} points.")

            st.session_state.last_calc_results = {
                'res_fine': res_fine,
                'res_optim_grid': res_optim_grid,
                'ep_used': ep_to_use.copy(),
                'stack_used': stack_calc,
                'materials_used': st.session_state.materials.copy(),
                'nSub_used': st.session_state.materials['Substrate'],
                'l0_used': l0_calc,
                'method_name': method_name,
            }

        except Exception as e:
            st.error(f"Une erreur inattendue est survenue pendant le calcul : {e}")
            add_log(f"ERREUR FATALE dans run_calculation_wrapper : {e}\n{traceback.format_exc(limit=3)}")
        finally:
             if 'Substrate' not in st.session_state.materials and 'nSub_material_calc' in locals():
                     st.session_state.materials['Substrate'] = nSub_material_calc

def undo_remove_wrapper():
    """Annule la dernière action de suppression de couche."""
    add_log("--- Annulation de la Dernière Suppression de Couche ---")
    if not st.session_state.get('ep_history'):
        st.warning("Aucune action à annuler.")
        return

    try:
        last_ep, last_stack = st.session_state.ep_history.pop()
        st.session_state.stack = last_stack
        st.session_state.current_ep = last_ep.copy()

        clear_optimized_state()

        add_log(f"Annulation réussie. Structure restaurée avec {len(last_stack)} couches.")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False,
            'method_name': "Annuler Suppression",
            'force_ep': last_ep.copy()
        }
    except IndexError:
        st.warning("L'historique d'annulation est vide.")
    except Exception as e:
        st.error(f"Erreur pendant l'opération d'annulation : {e}")
        add_log(f"ERREUR FATALE dans undo_remove_wrapper : {e}\n{traceback.format_exc(limit=2)}")

def run_de_optimization_wrapper(speed: str):
    """Exécute l'optimisation avec Differential Evolution, suivie d'un affinage local."""
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    clear_optimized_state()

    speed_to_maxiter = {
        "Rapide": 200,
        "Normal": 500,
        "Lent": 1000,
        "Très Lent": 2000
    }
    maxiter_de = speed_to_maxiter.get(speed, 2000) # Par défaut à Très Lent

    try:
        with st.spinner(f"Préparation de l'optimisation par Évolution Différentielle (Mode {speed})..."):
            try:
                num_layers_ui = len(st.session_state.stack)
                current_stack_from_ui = [{'material': st.session_state.get(f'mat_{i}'), 'qwot': st.session_state.get(f'qwot_{i}'), 'is_variable': st.session_state.get(f'var_{i}')} for i in range(num_layers_ui)]
                st.session_state.stack = current_stack_from_ui
                stack = current_stack_from_ui
            except KeyError as e:
                st.error(f"Erreur de synchronisation de l'état de l'interface. Clé manquante : {e}")
                return

            active_targets = validate_targets()
            if not active_targets:
                st.error("L'optimisation nécessite des cibles actives et valides.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None: return

            materials = st.session_state.materials.copy()
            nSub_material = materials.pop('Substrate')
            num_pts_for_optim = calculate_optim_points(active_targets)
            add_log(f"[Opt DE] Résolution d'optimisation : {num_pts_for_optim} points.")

            validated_inputs = {
                'l0': st.session_state.l0, 'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
                'materials': materials, 'nSub_material': nSub_material, 'stack': stack,
                'backside_enabled': st.session_state.get('backside_enabled', False),
                'backside_stack': st.session_state.backside_stack, 'backside_l0': st.session_state.backside_l0,
                'optim_resolution_points': num_pts_for_optim
            }

            if not stack:
                st.error("La structure avant est vide, impossible de démarrer l'optimisation.")
                return
            
            ep_start, logs_ep_init = calculate_initial_ep(stack, validated_inputs['l0'], materials)
            add_log(logs_ep_init)
            if ep_start is None: return

            num_layers = len(stack)
            variable_indices = [i for i, layer in enumerate(stack) if layer['is_variable']]
            fixed_indices = [i for i in range(num_layers) if not stack[i]['is_variable']]
            log_optimization_setup(stack, ep_start, variable_indices, fixed_indices)
            if not variable_indices:
                st.warning("Aucune couche variable à optimiser.")
                return

            l_vec_optim_np_base = np.linspace(l_min_opt, l_max_opt, num_pts_for_optim)
            target_points = []; [target_points.extend([t['min'], t['max']]) for t in active_targets]
            l_vec_optim_np = np.unique(np.concatenate([l_vec_optim_np_base, np.array(target_points)]))
            l_vec_optim_jax = jnp.asarray(l_vec_optim_np, dtype=DTYPE_FLOAT)
            material_arrays = {k: _get_nk_array_for_lambda_vec(p, l_vec_optim_jax)[0] for k,p in materials.items()}
            nSub_arr_optim, _ = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
            layer_indices_array = jnp.stack([material_arrays[layer['material']] for layer in stack], axis=0)
            active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max']), float(t.get('weight', 1.0))) for t in active_targets)
            
            backside_enabled = validated_inputs.get('backside_enabled', False)
            backside_ep_jnp = jnp.array([], dtype=DTYPE_FLOAT)
            backside_layer_indices_jnp = jnp.empty((0, len(l_vec_optim_jax)), dtype=DTYPE_COMPLEX)
            if backside_enabled and validated_inputs['backside_stack']:
                backside_ep, _ = calculate_initial_ep(validated_inputs['backside_stack'], validated_inputs['backside_l0'], materials)
                backside_ep_jnp = jnp.asarray(backside_ep, dtype=DTYPE_FLOAT)
                backside_layer_indices_jnp = jnp.stack([material_arrays[layer['material']] for layer in validated_inputs['backside_stack']], axis=0)

            cost_fn_jax = jax.jit(calculate_mse_for_optimization_penalized_jax)

            def scipy_obj_wrapper(ep_variable_np: np.ndarray, *args: Any) -> float:
                try:
                    full_ep_template_np, variable_indices_np, layer_indices_array_jax, nSub_arr_jax, l_vec_jax, active_targets_tup, min_thick_phys, backside_en, backside_ep_jax, backside_indices_jax = args
                    ep_full_jnp = jnp.array(full_ep_template_np).at[variable_indices_np].set(jnp.asarray(ep_variable_np))
                    value_jax = cost_fn_jax(ep_full_jnp, layer_indices_array_jax, nSub_arr_jax, l_vec_jax, active_targets_tup, min_thick_phys, backside_en, backside_ep_jax, backside_indices_jax)
                    value_float = np.array(value_jax, dtype=np.float64).item()
                    return value_float if np.isfinite(value_float) else np.inf
                except Exception: return np.inf

            full_ep_template = np.zeros(num_layers, dtype=np.float32)
            if fixed_indices: full_ep_template[fixed_indices] = ep_start[fixed_indices]
            static_args_for_scipy = (full_ep_template, np.array(variable_indices), layer_indices_array, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, MIN_THICKNESS_PHYS_NM, backside_enabled, backside_ep_jnp, backside_layer_indices_jnp)
            
            l0 = validated_inputs['l0']
            bounds = []
            action_radius = st.session_state.get('de_action_radius', 1.0) 
            add_log(f"[Opt DE] Rayon d'action utilisé : {action_radius}")

            for i in variable_indices:
                n_complex_at_l0, _ = _get_nk_at_lambda(materials[stack[i]['material']], l0)
                n_real_at_l0 = n_complex_at_l0.real if n_complex_at_l0 and n_complex_at_l0.real > 0 else 1.5
                upper_bound = action_radius * l0 / (2 * n_real_at_l0)
                bounds.append((0.0, upper_bound))
            
            add_log(f"[Opt DE] Démarrage de Differential Evolution (Mode: {speed}, maxiter={maxiter_de}).")
            
            with st.spinner(f"Optimisation Globale (Évolution Différentielle - {speed}) en cours..."):
                result = differential_evolution(
                    scipy_obj_wrapper,
                    bounds=bounds,
                    args=static_args_for_scipy,
                    maxiter=maxiter_de,
                    polish=False,
                    disp=False
                )

        if result.success and result.x is not None:
            final_ep_full = np.copy(ep_start)
            final_ep_full[variable_indices] = result.x
            final_ep_global = np.where(final_ep_full < MIN_THICKNESS_PHYS_NM, 0.0, final_ep_full)
            
            with st.spinner("Affinage du résultat avec une optimisation locale..."):
                add_log("[Opt Locale] Lancement de l'optimisation locale de finition...")
                final_ep_refined, success_refined, _, optim_logs_refined, msg_refined = \
                    _run_core_optimization(final_ep_global, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix="  [Affinage] ")
                add_log(optim_logs_refined)

                if success_refined and final_ep_refined is not None:
                    add_log("[Opt Locale] Affinage local réussi. Mise à jour de la structure.")
                    final_ep = final_ep_refined
                else:
                    add_log("[Opt Locale] AVERTISSEMENT : L'affinage local a échoué. Utilisation du résultat global.")
                    final_ep = final_ep_global

            update_succeeded = update_nominal_stack_from_ep(final_ep, stack, validated_inputs['l0'], materials)
            if update_succeeded:
                st.success(f"Optimisation (Évo. Diff. {speed} + Affinage) terminée. La structure a été mise à jour.")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {
                    'is_optimized_run': False,
                    'method_name': f"Nouveau Avant (depuis Opt DE {speed} + Affinage)",
                    'force_ep': final_ep.copy()
                }
        else:
            st.error(f"L'optimisation par Évolution Différentielle a échoué : {result.message}")

    except Exception as e:
        st.error(f"Erreur pendant l'optimisation par Évolution Différentielle : {e}")
        add_log(f"ERREUR FATALE (DE): {e}\n{traceback.format_exc(limit=2)}")
    finally:
        if 'Substrate' not in st.session_state.materials and 'nSub_material' in locals():
            st.session_state.materials['Substrate'] = nSub_material

def run_remove_thin_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    ep_start_removal = None
    stack_before_removal = st.session_state.stack.copy()

    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        ep_start_removal = st.session_state.optimized_ep.copy()
        stack_before_removal = st.session_state.get('optimized_stack_sequence', stack_before_removal)
    else:
        try:
            materials_temp = st.session_state.materials
            if not stack_before_removal:
                ep_start_removal = np.array([], dtype=np.float32)
            else:
                ep_start_removal, logs_ep_init = calculate_initial_ep(
                    stack_before_removal, st.session_state.l0, materials_temp
                )
                add_log(logs_ep_init)
                if ep_start_removal is None:
                    st.error("Échec du calcul de la structure nominale pour la suppression.")
                    return
            st.session_state.current_ep = ep_start_removal.copy()
        except Exception as e_nom:
            st.error(f"Erreur lors du calcul de la structure nominale pour la suppression : {e_nom}")
            return
    if ep_start_removal is None:
        st.error("Impossible de déterminer une structure de départ valide pour la suppression.")
        return
    if len(ep_start_removal) == 0:
        st.error("La structure est vide, impossible de supprimer des couches.")
        return

    with st.spinner("Suppression de la couche fine + Ré-optimisation..."):
        try:
            st.session_state.ep_history.append((ep_start_removal.copy(), stack_before_removal.copy()))

            ep_after_removal, new_stack, structure_changed, removal_logs = _perform_layer_removal(
                ep_start_removal, stack_before_removal, MIN_THICKNESS_PHYS_NM, log_prefix="  [Suppr.] "
            )
            add_log(removal_logs)

            if structure_changed and ep_after_removal is not None and new_stack is not None:
                add_log("Ré-optimisation après suppression...")

                active_targets = validate_targets()
                if active_targets is None or not active_targets:
                    st.error("Ré-optimisation annulée : cibles invalides ou manquantes.")
                    return

                l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
                if l_min_opt is None:
                    st.error("Ré-optimisation annulée : plage de longueurs d'onde invalide.")
                    return

                materials = st.session_state.materials.copy()
                nSub_material = materials.pop('Substrate')
                
                num_pts_for_reoptim = calculate_optim_points(active_targets)
                add_log(f"[RéOpt] Résolution de ré-optimisation calculée : {num_pts_for_reoptim} points.")

                validated_inputs = {
                    'l0': st.session_state.l0,
                    'auto_thin_threshold': st.session_state.auto_thin_threshold,
                    'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
                    'materials': materials,
                    'nSub_material': nSub_material,
                    'stack': new_stack, # Utiliser le nouveau stack pour l'optimisation
                    'backside_enabled': st.session_state.get('backside_enabled', False),
                    'backside_stack': st.session_state.backside_stack,
                    'backside_l0': st.session_state.backside_l0,
                    'optim_resolution_points': num_pts_for_reoptim
                }

                final_ep, success, final_cost, optim_logs, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix="  [RéOpt] ")
                add_log(optim_logs)

                if success and final_ep is not None:
                    update_succeeded = update_nominal_stack_from_ep(final_ep, new_stack, validated_inputs['l0'], materials)
                    if update_succeeded:
                        st.success(f"Suppression + Ré-optimisation terminées ({msg}). La structure a été mise à jour.")
                        st.session_state.needs_rerun_calc = True
                        st.session_state.rerun_calc_params = {
                            'is_optimized_run': False,
                            'method_name': f"Nouveau Avant (Post-Suppression)",
                            'force_ep': final_ep.copy()
                        }
                    else:
                        st.error("La ré-optimisation a réussi, mais la mise à jour de la structure a échoué.")
                else:
                    st.warning(f"Couche supprimée, mais la ré-optimisation a échoué ({msg}). La structure avant suppression est conservée.")
                    st.session_state.stack = stack_before_removal
            else:
                st.info("Aucune couche n'a été supprimée.")
                try:
                    st.session_state.ep_history.pop()
                except IndexError: pass
        except Exception as e:
            st.error(f"Erreur lors de la suppression de la couche fine : {e}")
            add_log(f"ERREUR : {e}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass

def run_monte_carlo_wrapper(container):
    with container:
        if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
            st.warning("Veuillez d'abord calculer une structure de base dans l'onglet 'Résultats'.")
            return

        with st.spinner("Lancement de la simulation Monte-Carlo..."):
            try:
                base_results = st.session_state.last_calc_results
                ep_base = base_results.get('ep_used')
                if ep_base is None or ep_base.size == 0:
                    st.error("Aucune structure de base à simuler. Veuillez d'abord évaluer une conception.")
                    return

                materials = base_results['materials_used']
                nSub_mat = base_results['nSub_used']
                stack = base_results['stack_used']
                l_vec = base_results['res_fine']['l']
                std_dev = st.session_state.monte_carlo_std_dev
                num_draws = 100
                active_targets = validate_targets()

                material_arrays = {}
                for key, props in materials.items():
                    arr, _ = _get_nk_array_for_lambda_vec(props, l_vec)
                    material_arrays[key] = arr

                nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec)

                indices_list = [material_arrays[layer['material']] for layer in stack]
                layer_indices_array = jnp.stack(indices_list, axis=0)

                @jax.jit
                def get_spectrum_for_one_ep(ep_vector, layer_indices, nSub, lambdas):
                    ep_vector_jnp = jnp.asarray(ep_vector)
                    layer_indices_T = layer_indices.T

                    Ts_arr_raw, _ = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                        lambdas, ep_vector_jnp, layer_indices_T, nSub
                    )
                    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
                    return jnp.clip(Ts_arr, 0.0, 1.0)

                vmap_calculate_T = jax.vmap(get_spectrum_for_one_ep, in_axes=(0, None, None, None))

                noise = np.random.normal(0, std_dev, (num_draws, len(ep_base))).astype(np.float32)
                perturbed_eps = ep_base + noise
                perturbed_eps = np.maximum(perturbed_eps, 0.0)

                all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), layer_indices_array, nSub_arr, l_vec))

                all_rmses = []
                if active_targets:
                    for i in range(num_draws):
                        res_temp = {'l': l_vec, 'Ts': all_ts_results[i]}
                        rmse, _ = calculate_final_rmse(res_temp, active_targets)
                        if rmse is not None:
                            all_rmses.append(rmse)

                plausible_rmse = np.percentile(all_rmses, 80) if all_rmses else None

                lower_bound = np.percentile(all_ts_results, 10, axis=0)
                upper_bound = np.percentile(all_ts_results, 90, axis=0)

                st.session_state.monte_carlo_results = {
                    'l_vec': l_vec,
                    'all_ts_results': all_ts_results,
                    'base_ts': base_results['res_fine']['Ts'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'std_dev': std_dev,
                    'plausible_rmse': plausible_rmse
                }
                add_log("Simulation de Monte-Carlo terminée.")

            except Exception as e:
                st.error(f"Une erreur est survenue pendant la simulation Monte-Carlo: {e}")
                add_log(f"ERREUR FATALE pendant Monte-Carlo : {e}")
                traceback.print_exc()

def run_tolerance_analysis_wrapper(container):
    with container:
        if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
            st.warning("Veuillez d'abord calculer une structure de base dans l'onglet 'Résultats'.")
            return

        with st.spinner("Lancement de l'analyse de tolérance..."):
            try:
                base_results = st.session_state.last_calc_results
                ep_base = base_results.get('ep_used')
                if ep_base is None or ep_base.size == 0:
                    st.error("Aucune structure de base à analyser.")
                    return

                materials = base_results['materials_used']
                nSub_mat = base_results['nSub_used']
                stack = base_results['stack_used']
                active_targets = validate_targets()
                if not active_targets:
                    st.error("L'analyse de tolérance nécessite des cibles actives pour calculer le RMSE.")
                    return

                l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                l_vec = np.geomspace(l_min_plot, l_max_plot, 100)

                std_devs_abs = np.linspace(0, 5, 15)
                std_devs_rel = np.linspace(0, 10, 15)
                num_draws = 100

                plausible_rmses_abs = []
                plausible_rmses_rel = []

                material_arrays = {}
                for key, props in materials.items():
                    arr, _ = _get_nk_array_for_lambda_vec(props, l_vec)
                    material_arrays[key] = arr

                nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec)

                indices_list = [material_arrays[layer['material']] for layer in stack]
                layer_indices_array = jnp.stack(indices_list, axis=0)

                @jax.jit
                def get_spectrum_for_one_ep(ep_vector, layer_indices, nSub, lambdas):
                    ep_vector_jnp = jnp.asarray(ep_vector)
                    layer_indices_T = layer_indices.T
                    Ts_arr_raw, _ = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                        lambdas, ep_vector_jnp, layer_indices_T, nSub
                    )
                    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
                    return jnp.clip(Ts_arr, 0.0, 1.0)

                vmap_calculate_T = jax.vmap(get_spectrum_for_one_ep, in_axes=(0, None, None, None))

                for std_dev in std_devs_abs:
                    noise = np.random.normal(0, std_dev, (num_draws, len(ep_base))).astype(np.float32)
                    perturbed_eps = ep_base + noise
                    perturbed_eps = np.maximum(perturbed_eps, 0.0)
                    all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), layer_indices_array, nSub_arr, l_vec))

                    all_rmses = [calculate_final_rmse({'l': l_vec, 'Ts': ts}, active_targets)[0] for ts in all_ts_results]
                    all_rmses = [r for r in all_rmses if r is not None]
                    plausible_rmses_abs.append(np.percentile(all_rmses, 80) if all_rmses else 0)

                for std_dev_percent in std_devs_rel:
                    std_dev_values = ep_base * (std_dev_percent / 100.0)
                    noise = np.random.normal(0, 1, (num_draws, len(ep_base))).astype(np.float32) * std_dev_values
                    perturbed_eps = ep_base + noise
                    perturbed_eps = np.maximum(perturbed_eps, 0.0)
                    all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), layer_indices_array, nSub_arr, l_vec))

                    all_rmses = [calculate_final_rmse({'l': l_vec, 'Ts': ts}, active_targets)[0] for ts in all_ts_results]
                    all_rmses = [r for r in all_rmses if r is not None]
                    plausible_rmses_rel.append(np.percentile(all_rmses, 80) if all_rmses else 0)

                st.session_state.tolerance_analysis_results = {
                    'std_devs_abs': std_devs_abs,
                    'plausible_rmses_abs': plausible_rmses_abs,
                    'std_devs_rel': std_devs_rel,
                    'plausible_rmses_rel': plausible_rmses_rel,
                }
                add_log("Analyse de tolérance terminée.")

            except Exception as e:
                st.error(f"Une erreur est survenue pendant l'analyse de tolérance: {e}")
                add_log(f"ERREUR FATALE pendant l'analyse de tolérance : {e}")
                traceback.print_exc()

def run_color_analysis_wrapper(container):
    """Exécute le calcul colorimétrique nominal et Monte-Carlo."""
    with container:
        if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
            st.warning("Veuillez d'abord calculer une structure de base dans l'onglet 'Résultats'.")
            return

        with st.spinner("Analyse colorimétrique en cours..."):
            try:
                base_results = st.session_state.last_calc_results
                ep_base = base_results.get('ep_used')
                if ep_base is None:
                    st.error("Aucune structure de base à analyser.")
                    return

                materials = base_results['materials_used']
                nSub_mat = base_results['nSub_used']
                stack = base_results['stack_used']
                backside_enabled = st.session_state.get('backside_enabled', False)
                backside_stack = st.session_state.backside_stack if backside_enabled else None
                backside_l0 = st.session_state.backside_l0 if backside_enabled else None
                add_log(f"[Analyse Couleur] Démarrage avec backside_enabled = {backside_enabled}")


                # --- 1. Calcul du point nominal ---
                res_nominal_color, logs_nom_color = calculate_TR_from_ep_jax(
                    ep_base, stack, materials, nSub_mat, CIE_LAMBDA,
                    backside_enabled=backside_enabled,
                    backside_stack=backside_stack,
                    backside_l0=backside_l0)
                add_log(logs_nom_color)
                if res_nominal_color is None:
                    st.error("Échec du calcul de la réflectance nominale pour la colorimétrie.")
                    return
                
                # Réflectance calculée comme 1-T, comme demandé
                reflectance_nominal = 1.0 - res_nominal_color['Ts']
                xyz_nominal = spectrum_to_xyz(res_nominal_color['l'], reflectance_nominal, CIE_LAMBDA, CIE_X, CIE_Y, CIE_Z, ILLUMINANT_D65)
                lab_nominal = xyz_to_lab(xyz_nominal, XYZ_N_D65)

                # Calcul des réflectances moyennes
                r_moy_simple = np.mean(reflectance_nominal)
                r_moy_photopic = np.sum(reflectance_nominal * CIE_Y) / np.sum(CIE_Y)


                # --- 2. Préparation pour la simulation Monte-Carlo ---
                num_draws = 200
                sigma_nm = 2.0
                
                l_vec_color_jnp = jnp.array(CIE_LAMBDA, dtype=DTYPE_FLOAT)
                material_arrays_color = {key: _get_nk_array_for_lambda_vec(props, l_vec_color_jnp)[0] for key, props in materials.items()}
                nSub_arr_color, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_color_jnp)
                
                front_layer_indices_array_color = jnp.stack([material_arrays_color[layer['material']] for layer in stack], axis=0)

                backside_ep_jnp = jnp.array([], dtype=DTYPE_FLOAT)
                backside_layer_indices_array_color = jnp.empty((0, len(l_vec_color_jnp)), dtype=DTYPE_COMPLEX)
                if backside_enabled and backside_stack:
                    back_ep, _ = calculate_initial_ep(backside_stack, backside_l0, materials)
                    if back_ep is not None:
                        backside_ep_jnp = jnp.array(back_ep)
                        back_indices_list_color = [material_arrays_color[layer['material']] for layer in backside_stack]
                        backside_layer_indices_array_color = jnp.stack(back_indices_list_color, axis=0)

                # --- Fonctions JIT spécialisées ---
                @jax.jit
                def get_transmittance_spectrum_1_face(ep_vector):
                    Ta_raw, _ = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                        l_vec_color_jnp, ep_vector, front_layer_indices_array_color.T, nSub_arr_color
                    )
                    return jnp.clip(jnp.nan_to_num(Ta_raw, nan=0.0), 0.0, 1.0)

                @jax.jit
                def get_transmittance_spectrum_2_faces(ep_vector):
                    Ta_raw, Ra_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                        l_vec_color_jnp, ep_vector, front_layer_indices_array_color.T, nSub_arr_color
                    )
                    Ta = jnp.clip(jnp.nan_to_num(Ta_raw, nan=0.0), 0.0, 1.0)
                    Ra = jnp.clip(jnp.nan_to_num(Ra_raw, nan=0.0), 0.0, 1.0)
                    
                    Tb_raw, Rb_raw = vmap(calculate_single_wavelength_TR_core, in_axes=(0, None, 0, 0))(
                        l_vec_color_jnp, backside_ep_jnp, backside_layer_indices_array_color.T, nSub_arr_color
                    )
                    Tb = jnp.clip(jnp.nan_to_num(Tb_raw, nan=0.0), 0.0, 1.0)
                    Rb = jnp.clip(jnp.nan_to_num(Rb_raw, nan=0.0), 0.0, 1.0)

                    denominator = 1.0 - Ra * Rb
                    safe_denominator = jnp.where(denominator < 1e-9, 1e-9, denominator)
                    T_total = (Ta * Tb) / safe_denominator
                    return jnp.clip(T_total, 0.0, 1.0)

                # --- Sélection de la fonction et vmap ---
                if backside_enabled:
                    vmap_get_transmittance = jax.vmap(get_transmittance_spectrum_2_faces)
                else:
                    vmap_get_transmittance = jax.vmap(get_transmittance_spectrum_1_face)
                
                noise = np.random.normal(0, sigma_nm, (num_draws, len(ep_base))).astype(np.float32)
                perturbed_eps = jnp.maximum(ep_base + noise, 0.0)
                
                all_ts_results = np.array(vmap_get_transmittance(perturbed_eps))
                all_rs_results = 1.0 - all_ts_results
                
                # --- 3. Conversion de tous les spectres en L*a*b* ---
                lab_points = [xyz_to_lab(spectrum_to_xyz(CIE_LAMBDA, rs, CIE_LAMBDA, CIE_X, CIE_Y, CIE_Z, ILLUMINANT_D65), XYZ_N_D65) for rs in all_rs_results]
                
                # --- 4. Stockage des résultats ---
                st.session_state.color_results = {
                    'lab_nominal': lab_nominal,
                    'lab_mc_points': np.array(lab_points),
                    'r_moy_simple': r_moy_simple,
                    'r_moy_photopic': r_moy_photopic,
                    'backside_mode': "2 faces" if backside_enabled else "1 face"
                }
                add_log(f"Analyse colorimétrique terminée. Point nominal (L*,a*,b*): {lab_nominal[0]:.2f}, {lab_nominal[1]:.2f}, {lab_nominal[2]:.2f}")

            except Exception as e:
                st.error(f"Une erreur est survenue pendant l'analyse colorimétrique : {e}")
                add_log(f"ERREUR FATALE pendant l'analyse couleur : {e}\n{traceback.format_exc(limit=3)}")

# --- SECTION : DÉMARRAGE DE L'APPLICATION ET INTERFACE PRINCIPALE ---

st.set_page_config(layout="wide", page_title="formation_CMO_2025")

# Initialisation unique de l'état de la session
initialize_session_state()

def update_stack_item(index, field):
    """Callback pour mettre à jour un champ spécifique d'une couche dans st.session_state.stack."""
    widget_key = f"{field}_{index}"
    if widget_key in st.session_state:
        st.session_state.stack[index][field] = st.session_state[widget_key]
        trigger_nominal_recalc()

def update_target_item(index, field):
    """Callback pour mettre à jour un champ de cible et déclencher un recalcul si nécessaire."""
    widget_key = f"target_{field}_{index}"
    if widget_key in st.session_state:
        st.session_state.targets[index][field] = st.session_state[widget_key]
        if field != 'enabled':
             trigger_nominal_recalc()

def init_qwots():
    """Réinitialise tous les QWOTs de la structure avant à 1.0."""
    for layer in st.session_state.stack:
        layer['qwot'] = 1.0
    trigger_nominal_recalc()
    st.rerun()

def sync_backside_checkboxes(source_key):
    """Synchronise l'état de la case à cocher 'backside_enabled'."""
    st.session_state.backside_enabled = st.session_state[source_key]
    trigger_nominal_recalc()

st.title("Formation CMO 2025")
st.divider()

main_layout = st.columns([1, 3])

with main_layout[0]:
    st.subheader("Matériaux (Modèle de Cauchy)")

    for mat_key, mat_props in st.session_state.materials.items():
        with st.expander(f"Matériau {mat_key}"):
            cols = st.columns(2)
            n400_key = f"{mat_key}_400"
            n700_key = f"{mat_key}_700"
            mat_props['n_400'] = cols[0].number_input("Indice @ 400nm", value=mat_props['n_400'], format="%.4f", key=n400_key, on_change=trigger_nominal_recalc, help="Indice de réfraction du matériau à la longueur d'onde de 400 nm.")
            mat_props['n_700'] = cols[1].number_input("Indice @ 700nm", value=mat_props['n_700'], format="%.4f", key=n700_key, on_change=trigger_nominal_recalc, help="Indice de réfraction du matériau à la longueur d'onde de 700 nm.")

    st.subheader("Structure Avant")
    st.number_input("Longueur d'onde de référence λ₀ (nm) pour QWOT", min_value=1.0, format="%.2f", key="l0", on_change=trigger_nominal_recalc, help="Longueur d'onde centrale utilisée pour calculer les épaisseurs physiques à partir des QWOT (Quart d'Onde Optique).")

    layer_materials = st.session_state.materials.copy()
    layer_materials.pop('Substrate', None)
    
    ep_vector_display, logs_ep_disp = calculate_initial_ep(st.session_state.stack, st.session_state.l0, layer_materials)
    if ep_vector_display is None:
        ep_vector_display = np.full(len(st.session_state.stack), np.nan)
        add_log(logs_ep_disp)

    cols = st.columns([1, 2, 2, 2, 1])
    cols[0].markdown("**Couche**")
    cols[1].markdown("**Matériau**")
    cols[2].markdown("**Épaisseur (QWOT)**")
    cols[3].markdown("<div style='color:grey'>Ép. Phys (nm)</div>", unsafe_allow_html=True)
    cols[4].markdown("**Var.**")

    for i, layer in enumerate(st.session_state.stack):
        cols = st.columns([1, 2, 2, 2, 1])
        cols[0].write(f"&nbsp;&nbsp;&nbsp;{i + 1}")
        cols[1].selectbox(f"Matériau Couche {i+1}", options=MATERIAL_KEYS, index=MATERIAL_KEYS.index(layer['material']), key=f"mat_{i}", label_visibility="collapsed", on_change=update_stack_item, args=(i, 'material'), help="Choisissez le matériau pour cette couche.")
        cols[2].number_input(f"QWOT Couche {i+1}", value=float(layer['qwot']), key=f"qwot_{i}", label_visibility="collapsed", min_value=0.0, format="%.4f", on_change=update_stack_item, args=(i, 'qwot'), help="Épaisseur optique de la couche, exprimée en multiples de quart d'onde (QWOT). 1.0 = λ₀ / (4n).")
        ep_phys_display = "N/A"
        if i < len(ep_vector_display) and not np.isnan(ep_vector_display[i]):
            ep_phys_display = f"{ep_vector_display[i]:.1f}"
        cols[3].text_input(f"Ep Phys {i+1}", value=ep_phys_display, key=f"ep_phys_{i}", disabled=True, label_visibility="collapsed")
        cols[4].checkbox("Var", value=layer['is_variable'], key=f"var_{i}", label_visibility="collapsed", on_change=update_stack_item, args=(i, 'is_variable'), help="Cochez si l'épaisseur de cette couche doit être modifiée par l'optimiseur.")

    cols = st.columns(3)
    if cols[0].button("Ajouter une couche", use_container_width=True, disabled=len(st.session_state.stack) >= MAX_LAYERS, help="Ajoute une nouvelle couche à la fin de la structure."):
        if len(st.session_state.stack) < MAX_LAYERS:
            last_material = st.session_state.stack[-1]['material'] if st.session_state.stack else 'L'
            new_material = 'L' if last_material == 'H' else 'H'
            st.session_state.stack.append({'material': new_material, 'qwot': 1.0, 'is_variable': True})
            trigger_nominal_recalc(); st.rerun()
    if cols[1].button("Supprimer la dernière", use_container_width=True, disabled=len(st.session_state.stack) == 0, help="Supprime la dernière couche de la structure."):
        if st.session_state.stack:
            st.session_state.stack.pop()
            trigger_nominal_recalc(); st.rerun()
    if cols[2].button("Initialiser QWOTs", use_container_width=True, on_click=init_qwots, help="Réinitialise toutes les épaisseurs QWOT à 1.0"): pass

    st.caption(f"Nombre de couches : {len(st.session_state.stack)} / {MAX_LAYERS}")
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        ep_display = st.session_state.last_calc_results.get('ep_used')
        if ep_display is not None and ep_display.size > 0:
            st.markdown("**Épaisseurs physiques (nm)**")
            formatted_thicknesses = [f"<span style='color:blue;'>{i + 1}.</span> {t:.1f}" for i, t in enumerate(ep_display)]
            st.markdown(", ".join(formatted_thicknesses), unsafe_allow_html=True)

    st.subheader("Cibles & Paramètres")
    st.checkbox("Échelle Y automatique", key="auto_scale_y", help="Si coché, l'axe Y du graphique de transmittance s'ajuste automatiquement aux valeurs min/max. Sinon, il est fixé de -0.05 à 1.05.", on_change=trigger_nominal_recalc)
    hdr_cols = st.columns([0.5, 1, 1, 1, 1, 0.8])
    hdrs = ["On", "λmin", "λmax", "Tmin", "Tmax", "Poids"]
    for c, h in zip(hdr_cols, hdrs): c.caption(h)
    for i in range(3):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1, 1, 1, 1, 0.8])
        cols[0].checkbox(f"Cible {i+1} activée", value=target.get('enabled', False), key=f"target_enabled_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'enabled'), help=f"Activer/Désactiver la cible {i+1}")
        cols[1].number_input(f"λmin Cible {i+1}", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'min'), help="Longueur d'onde minimale de la cible (nm)")
        cols[2].number_input(f"λmax Cible {i+1}", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'max'), help="Longueur d'onde maximale de la cible (nm)")
        cols[3].number_input(f"Tmin Cible {i+1}", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_target_min_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'target_min'), help="Transmittance cible à λmin (0=0%, 1=100%)")
        cols[4].number_input(f"Tmax Cible {i+1}", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_target_max_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'target_max'), help="Transmittance cible à λmax (0=0%, 1=100%)")
        cols[5].number_input(f"Poids Cible {i+1}", value=target.get('weight', 1.0), min_value=0.0, format="%.2f", step=0.1, key=f"target_weight_{i}", label_visibility="collapsed", on_change=update_target_item, args=(i, 'weight'), help="Importance relative de cette cible dans le calcul d'erreur (RMSE).")

with main_layout[1]:
    results_tab, indices_tab, color_tab, backside_tab, random_draws_tab, tolerance_tab, logs_tab = st.tabs([
        "**Résultats**",
        "**Tracé d'indices**", 
        "**Rendu Colorimétrique**",
        "**Face Arrière**", 
        "**Tirages Aléatoires**", 
        "**Analyse de Tolérance**", 
        "**Logs**"
    ])

    with results_tab:
        st.subheader("Actions")
        menu_cols = st.columns(4)
        with menu_cols[0]:
            if st.button("📊 Éval. Avant", key="eval_front_top", help="Calcule et affiche le spectre de la structure avant définie à gauche.", use_container_width=True):
                st.session_state.action = 'eval_front'
        with menu_cols[1]:
            if st.button("🧬 Opt. Globale", key="optim_de_top", help="Lance une optimisation globale (Évolution Différentielle) suivie d'un affinage local pour trouver la meilleure structure respectant les cibles.", use_container_width=True):
                st.session_state.action = 'opt_de'
        with menu_cols[2]:
            can_remove_structurally = st.session_state.get('current_ep') is not None and len(st.session_state.get('current_ep')) > 2
            if st.button("🗑️ Suppr.+RéOpt", key="remove_thin_top", help="Identifie la couche la plus fine, la supprime (ou fusionne si possible), puis relance une optimisation locale.", disabled=not can_remove_structurally, use_container_width=True):
                st.session_state.action = 'remove_thin'
        with menu_cols[3]:
            can_undo_top = bool(st.session_state.get('ep_history'))
            if st.button(f"↩️ Annuler ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_top", help="Annule la dernière action de suppression de couche.", disabled=not can_undo_top, use_container_width=True):
                undo_remove_wrapper(); st.rerun()
        
        st.slider(
            "Rayon d'action de l'optimisation globale",
            min_value=0.8, max_value=1.5, value=st.session_state.get('de_action_radius', 1.0), step=0.1,
            key='de_action_radius',
            help="Définit la plage de recherche pour l'optimisation globale. Un rayon plus grand explore des épaisseurs plus variées mais peut être plus lent."
        )

        st.divider()
        if st.session_state.get('last_calc_results'):
            st.subheader("Résultats Finaux")
            results_data = st.session_state.last_calc_results
            state_desc = "Optimisée" if st.session_state.is_optimized_state else "Structure Avant"
            ep_display = results_data.get('ep_used')
            num_layers_display = len(ep_display) if ep_display is not None else 0
            res_info_cols = st.columns(3)
            res_info_cols[0].caption(f"État : {state_desc} ({num_layers_display} couches)")
            res_info_cols[1].metric(label="RMSE (Pondéré)", value=f"{st.session_state.last_rmse:.4e}" if st.session_state.last_rmse is not None else "N/A")
            min_thick_str = "N/A"
            if ep_display is not None and ep_display.size > 0:
                valid_thick = ep_display[ep_display > 1e-9]
                if valid_thick.size > 0: min_thick_str = f"{np.min(valid_thick):.3f} nm"
            res_info_cols[2].caption(f"Ép. Min : {min_thick_str}")
            if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
                st.text_input("QWOT Optimisé (à λ₀ d'origine)", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_res")
            res_fine_plot, active_targets_plot, rmse_plot, res_optim_grid_plot = results_data.get('res_fine'), validate_targets(), st.session_state.last_rmse, results_data.get('res_optim_grid')
            if res_fine_plot and active_targets_plot is not None:
                fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
                fig_spec.suptitle(f'Transmission ({"2 faces" if st.session_state.get("backside_enabled") else "1 face"})', fontsize=12, weight='bold')
                try:
                    if 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                        ax_spec.plot(np.asarray(res_fine_plot['l']), np.asarray(res_fine_plot['Ts']), label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
                    plotted_target_label = False
                    if res_optim_grid_plot and 'l' in res_optim_grid_plot and res_optim_grid_plot['l'].size > 0:
                        optim_lambdas = np.asarray(res_optim_grid_plot['l'])
                        if active_targets_plot:
                            for target in active_targets_plot:
                                l_min, l_max, t_min, t_max_corr = target['min'], target['max'], target['target_min'], target['target_max']
                                indices_in_range = np.where((optim_lambdas >= l_min) & (optim_lambdas <= l_max))[0]
                                if indices_in_range.size > 0:
                                    target_x = optim_lambdas[indices_in_range]
                                    target_y = t_min + ((t_max_corr - t_min) / (l_max - l_min) if abs(l_max - l_min) > 1e-9 else 0.0) * (target_x - l_min)
                                    ax_spec.plot(target_x, target_y, 'x', color='red', markersize=5, label='Cible(s)' if not plotted_target_label else "_nolegend_", zorder=6)
                                    plotted_target_label = True
                    ax_spec.set_xlabel("Longueur d'onde (nm)"); ax_spec.set_ylabel('Transmittance'); ax_spec.grid(True, which='both', linestyle=':'); ax_spec.minorticks_on()
                    if 'res_l_plot' in locals() and len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                    if not st.session_state.get('auto_scale_y', False): ax_spec.set_ylim(-0.05, 1.05)
                    ax_spec.legend(fontsize=8)
                    ax_spec.text(0.98, 0.98, f"RMSE = {rmse_plot:.3e}" if rmse_plot is not None else "RMSE: N/A", transform=ax_spec.transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
                except Exception as e: ax_spec.text(0.5, 0.5, f"Erreur de tracé:\n{e}", ha='center', va='center', color='red')
                plt.tight_layout(rect=[0, 0, 1, 0.93]); st.pyplot(fig_spec); plt.close(fig_spec)
            plot_col1, plot_col2 = st.columns(2)
            ep_plot, l0_plot, materials_plot, nSub_plot, stack_plot = results_data.get('ep_used'), results_data.get('l0_used'), results_data.get('materials_used'), results_data.get('nSub_used'), results_data.get('stack_used')
            if all(v is not None for v in [ep_plot, l0_plot, materials_plot, nSub_plot, stack_plot]):
                with plot_col1:
                    fig_idx, ax_idx = plt.subplots(figsize=(6, 4))
                    try:
                        n_reals = [ _get_nk_at_lambda(materials_plot[l['material']], l0_plot)[0].real for l in stack_plot ]
                        n_sub = _get_nk_at_lambda(nSub_plot, l0_plot)[0].real
                        x, y = [-max(50, 0.1*ep_plot.sum())], [n_sub]
                        if len(ep_plot) > 0:
                            ep_cum = np.cumsum(ep_plot)
                            x.extend([0,0]); y.extend([n_sub, n_reals[0]])
                            for i in range(len(ep_plot)-1): x.extend([ep_cum[i], ep_cum[i]]); y.extend([n_reals[i], n_reals[i+1]])
                            x.extend([ep_cum[-1], ep_cum[-1], ep_cum[-1]+max(50, 0.1*ep_plot.sum())]); y.extend([n_reals[-1], 1.0, 1.0])
                        else: x.extend([0,max(50, 0.1*ep_plot.sum())]); y.extend([1.0, 1.0])
                        ax_idx.plot(x, y); ax_idx.set_xlabel('Profondeur (nm)'); ax_idx.set_ylabel("Indice"); ax_idx.set_title(f"Profil d'indice (à λ={l0_plot:.0f}nm)", fontsize=10); ax_idx.grid(True, linestyle=':')
                    except Exception as e: ax_idx.text(0.5, 0.5, f"Erreur:\n{e}", ha='center', va='center')
                    plt.tight_layout(); st.pyplot(fig_idx); plt.close(fig_idx)
                with plot_col2:
                    fig_stack, ax_stack = plt.subplots(figsize=(6, 4))
                    try:
                        if len(ep_plot) > 0:
                            layer_types = [l['material'] for l in stack_plot]; cmap = plt.colormaps.get_cmap('viridis'); colors = [cmap(i) for i in np.linspace(0, 1, len(MATERIAL_KEYS))]; color_map = {mat: colors[i] for i, mat in enumerate(MATERIAL_KEYS)}; layer_colors = [color_map[mat] for mat in layer_types]
                            ax_stack.barh(np.arange(len(ep_plot)), ep_plot, align='center', color=layer_colors, edgecolor='grey')
                            ax_stack.set_yticks(np.arange(len(ep_plot))); ax_stack.set_yticklabels([f"C{i+1} ({t})" for i, t in enumerate(layer_types)], fontsize=7); ax_stack.invert_yaxis()
                        else: ax_stack.text(0.5, 0.5, "Vide", ha='center')
                        ax_stack.set_xlabel('Épaisseur (nm)'); ax_stack.set_title(f"Structure ({len(ep_plot)} couches)", fontsize=10)
                    except Exception as e: ax_stack.text(0.5, 0.5, f"Erreur:\n{e}", ha='center')
                    plt.tight_layout(); st.pyplot(fig_stack); plt.close(fig_stack)

    with indices_tab:
        st.subheader("Indices de réfraction des matériaux")
        try:
            fig_indices, ax_indices = plt.subplots(figsize=(12, 5))
            l_vec_plot = jnp.linspace(350, 1000, 200)
            for mat_key, mat_props in st.session_state.materials.items():
                nk_array, _ = _get_nk_array_for_lambda_vec(mat_props, l_vec_plot)
                if nk_array is not None:
                    ax_indices.plot(l_vec_plot, nk_array.real, label=f"{mat_key}")
                else:
                    st.warning(f"Impossible de tracer l'indice pour le matériau '{mat_key}'.")
            ax_indices.set_title("Dispersion des indices de réfraction")
            ax_indices.set_xlabel("Longueur d'onde (nm)")
            ax_indices.set_ylabel("Indice de réfraction (n)")
            ax_indices.grid(True, which='both', linestyle='--')
            ax_indices.legend()
            ax_indices.set_xlim(350, 1000)
            st.pyplot(fig_indices)
            plt.close(fig_indices)
        except Exception as e:
            st.error(f"Une erreur est survenue lors du tracé des indices : {e}")

    with color_tab:
        st.subheader("Analyse du Rendu Colorimétrique (CIELAB)")
        # Ajout de la checkbox synchronisée
        st.checkbox(
            "Prendre en compte la face arrière",
            value=st.session_state.backside_enabled,
            key="backside_enabled_color_tab",
            on_change=sync_backside_checkboxes,
            args=("backside_enabled_color_tab",),
            help="Inclut les réflexions de la face arrière du substrat dans le calcul. Synchronisé avec l'onglet 'Face Arrière'."
        )
        st.info("Cette analyse calcule la couleur de la réflectance (calculée comme 1-T) en utilisant l'illuminant D65 et l'observateur standard CIE 1931 à 2 degrés.")
        if st.button("🎨 Lancer l'analyse colorimétrique", key="run_color", help="Calcule le point de couleur nominal et une simulation de Monte-Carlo (200 tirages, σ=2nm) pour visualiser la dispersion des couleurs."):
            st.session_state.action = 'color_analysis'
            st.rerun()

        if 'color_results' in st.session_state and st.session_state.color_results:
            color_data = st.session_state.color_results
            lab_nominal = color_data['lab_nominal']
            lab_mc = color_data['lab_mc_points']

            st.caption(f"Configuration du calcul : **{color_data['backside_mode']}**")

            col1, col2 = st.columns([2,1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                plot_cielab_diagram(ax, lab_nominal, lab_mc)
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                st.metric("L* (Luminance)", f"{lab_nominal[0]:.2f}", help="L'axe de la clarté. 0 = noir, 100 = blanc.")
                st.metric("R moy. Photopique (Y)", f"{color_data['r_moy_photopic']:.2%}", help="Réflectance moyenne pondérée par la sensibilité de l'œil V(λ). Liée à L*.")
                st.metric("R moy. (380-780nm)", f"{color_data['r_moy_simple']:.2%}", help="Réflectance moyenne arithmétique sur la gamme visible.")
                st.divider()
                st.metric("a*", f"{lab_nominal[1]:.2f}", help="L'axe rouge-vert. a* > 0 est rouge, a* < 0 est vert.")
                st.metric("b*", f"{lab_nominal[2]:.2f}", help="L'axe jaune-bleu. b* > 0 est jaune, b* < 0 est bleu.")
    
    with backside_tab:
        st.subheader("Structure Face Arrière (Épaisseurs Fixes)")
        st.checkbox(
            "Prendre en compte la face arrière",
            value=st.session_state.backside_enabled,
            key="backside_enabled_backside_tab",
            on_change=sync_backside_checkboxes,
            args=("backside_enabled_backside_tab",),
            help="Inclut les réflexions de la face arrière du substrat dans le calcul. Utile pour les substrats épais et transparents."
        )
        st.number_input("λ₀ de référence (nm) pour QWOT Face Arrière", min_value=1.0, format="%.2f", key="backside_l0", on_change=trigger_nominal_recalc, help="Longueur d'onde de référence pour la structure sur la face arrière.")
        def update_backside_stack_item(index, field):
            widget_key = f"back_{field}_{index}"
            if widget_key in st.session_state:
                st.session_state.backside_stack[index][field] = st.session_state[widget_key]
                trigger_nominal_recalc()
        layer_materials_back = st.session_state.materials.copy(); layer_materials_back.pop('Substrate', None)
        ep_vector_back_display, _ = calculate_initial_ep(st.session_state.backside_stack, st.session_state.backside_l0, layer_materials_back)
        if ep_vector_back_display is None: ep_vector_back_display = np.full(len(st.session_state.backside_stack), np.nan)
        cols = st.columns([1, 2, 2, 2]); cols[0].markdown("**Couche**"); cols[1].markdown("**Matériau**"); cols[2].markdown("**QWOT**"); cols[3].markdown("<div style='color:grey'>Ép. Phys (nm)</div>", unsafe_allow_html=True)
        for i, layer in enumerate(st.session_state.backside_stack):
            cols = st.columns([1, 2, 2, 2]); cols[0].write(f"&nbsp;&nbsp;&nbsp;{i + 1}")
            cols[1].selectbox(f"Mat Back {i+1}", options=MATERIAL_KEYS, index=MATERIAL_KEYS.index(layer['material']), key=f"back_mat_{i}", label_visibility="collapsed", on_change=update_backside_stack_item, args=(i, 'material'))
            cols[2].number_input(f"QWOT Back {i+1}", value=float(layer['qwot']), key=f"back_qwot_{i}", label_visibility="collapsed", min_value=0.0, format="%.4f", on_change=update_backside_stack_item, args=(i, 'qwot'))
            ep_phys_display_back = f"{ep_vector_back_display[i]:.1f}" if i < len(ep_vector_back_display) and not np.isnan(ep_vector_back_display[i]) else "N/A"
            cols[3].text_input(f"Ep Phys Back {i+1}", value=ep_phys_display_back, key=f"ep_phys_back_{i}", disabled=True, label_visibility="collapsed")
        cols = st.columns(2)
        if cols[0].button("Ajouter (Face Arrière)", use_container_width=True, disabled=len(st.session_state.backside_stack) >= MAX_LAYERS, help="Ajoute une couche à la structure de la face arrière."):
            if len(st.session_state.backside_stack) < MAX_LAYERS:
                last_mat = st.session_state.backside_stack[-1]['material'] if st.session_state.backside_stack else 'L'
                st.session_state.backside_stack.append({'material': 'L' if last_mat == 'H' else 'H', 'qwot': 1.0})
                trigger_nominal_recalc(); st.rerun()
        if cols[1].button("Supprimer (Face Arrière)", use_container_width=True, disabled=len(st.session_state.backside_stack) == 0, help="Supprime la dernière couche de la structure de la face arrière."):
            if st.session_state.backside_stack: st.session_state.backside_stack.pop(); trigger_nominal_recalc(); st.rerun()

    with random_draws_tab:
        st.subheader("Simulation de Monte-Carlo")
        st.number_input("Écart-type pour l'épaisseur (nm)", min_value=0.0, step=0.1, format="%.2f", key="monte_carlo_std_dev", help="Écart-type (en nm) de l'erreur aléatoire (distribution normale) appliquée à chaque couche pour la simulation.")
        if st.button("Lancer la simulation", key="run_mc", help="Simule 100 variations de la structure avec des erreurs aléatoires sur les épaisseurs pour évaluer la robustesse de la conception."): st.session_state.action = 'monte_carlo'; st.rerun()
        if 'monte_carlo_results' in st.session_state and st.session_state.monte_carlo_results:
            mc_data = st.session_state.monte_carlo_results
            if mc_data.get('plausible_rmse') is not None: st.metric(label="RMSE Plausible (80% des cas)", value=f"{mc_data['plausible_rmse']:.4e}")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(mc_data['l_vec'], mc_data['base_ts'], color='red', lw=2, label='Idéal', zorder=3)
            ax.scatter(np.tile(mc_data['l_vec'], 100), mc_data['all_ts_results'].flatten(), color='lightgray', alpha=0.2, s=2, zorder=1)
            ax.fill_between(mc_data['l_vec'], mc_data['lower_bound'], mc_data['upper_bound'], color='blue', alpha=0.3, label='Intervalle 80%', zorder=2)
            ax.set_xlabel("Longueur d'onde (nm)"); ax.set_ylabel("Transmittance"); ax.set_title(f"Monte-Carlo (100 tirages, σ={mc_data['std_dev']} nm)"); ax.legend(); ax.grid(True, linestyle=':')
            if not st.session_state.get('auto_scale_y', False): ax.set_ylim(-0.05, 1.05)
            st.pyplot(fig); plt.close(fig)

    with tolerance_tab:
        st.subheader("Analyse de Tolérance")
        if st.button("Lancer l'analyse de tolérance", key="run_tolerance", help="Calcule l'impact de différentes erreurs de fabrication (en nm absolus et en % relatifs) sur la performance (RMSE)."): st.session_state.action = 'tolerance_analysis'; st.rerun()
        if 'tolerance_analysis_results' in st.session_state and st.session_state.tolerance_analysis_results:
            tol_data = st.session_state.tolerance_analysis_results
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.set_xlabel('Écart-type absolu (nm)', color='tab:blue'); ax1.set_ylabel('RMSE Plausible (80%)', color='tab:blue'); ax1.plot(tol_data['std_devs_abs'], tol_data['plausible_rmses_abs'], color='tab:blue', marker='o', lw=2.5); ax1.tick_params(axis='x', labelcolor='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.grid(True, linestyle=':', color='tab:blue', alpha=0.5)
            ax2 = ax1.twiny(); ax2.set_xlabel('Écart-type relatif (%)', color='tab:orange'); ax2.plot(tol_data['std_devs_rel'], tol_data['plausible_rmses_rel'], color='tab:orange', marker='x', lw=2.5); ax2.tick_params(axis='x', labelcolor='tab:orange')
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with logs_tab:
        st.subheader("Logs")
        if st.button("Effacer les logs", help="Nettoie la fenêtre des logs."): st.session_state.log_messages.clear(); st.rerun()
        log_text = "\n".join(st.session_state.get('log_messages', ['Aucun log.']))
        st.code(log_text, language='text')

# --- SECTION : BOUCLE PRINCIPALE DE CONTRÔLE ---

action_to_run = st.session_state.get('action')
if action_to_run:
    st.session_state.action = None
    if action_to_run == 'eval_front':
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Structure Avant (Évaluée)"}
    elif action_to_run == 'opt_de': run_de_optimization_wrapper(speed="Très Lent")
    elif action_to_run == 'remove_thin': run_remove_thin_wrapper()
    elif action_to_run == 'monte_carlo': run_monte_carlo_wrapper(random_draws_tab)
    elif action_to_run == 'tolerance_analysis': run_tolerance_analysis_wrapper(tolerance_tab)
    elif action_to_run == 'color_analysis': run_color_analysis_wrapper(color_tab)
    st.rerun()

if st.session_state.get('needs_rerun_calc', False):
    params = st.session_state.rerun_calc_params
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True
    run_calculation_wrapper(is_optimized_run=params.get('is_optimized_run', False), method_name=params.get('method_name', 'Recalcul auto.'), force_ep=params.get('force_ep'))
    st.session_state.calculating = False
    st.rerun()
