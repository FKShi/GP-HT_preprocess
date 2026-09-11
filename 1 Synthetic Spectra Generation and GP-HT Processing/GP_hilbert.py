# -*- coding: utf-8 -*-
"""GP-HT DRT kernel and posterior-prediction functions from the companion code of the original GP-HT work.
This module is a shared function file. Keep the input data, model formulas, fitting objectives, and output formats consistent with the notebook calls.
When using GP-HT companion functions, the theoretical source is the original GP-HT work by Ciucci et al. (J. Electrochem. Soc. 2020, DOI: 10.1149/1945-7111/aba9c0).
"""
from math import pi, sqrt
from scipy.special import dawsn
from scipy.optimize import minimize
import numpy as np
def is_PD(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
# Matrix dimensions and matrix operations
def nearest_PD(A):
    B = (A + A.T)/2
    _, Sigma_mat, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))
    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2
    k = 1
    I = np.eye(A_symm.shape[0])
    while not is_PD(A_symm):
        eps = np.spacing(np.linalg.norm(A_symm))
        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1
    return A_symm
# Kernel-function calculation
# Kernel-function calculation
# Frequency-limitation or endpoint-truncation settings
def k_DRT(omega, omega_prime, sigma_DRT, tau_max, block):
    if block == 're':
        # Kernel-function calculation
        if omega == omega_prime:
            out_val = 0.5*(tau_max/(1+(tau_max*omega)**2)\
                            + np.arctan(tau_max*omega)/omega)
        else:
            num_out_val = omega*np.arctan(tau_max*omega)\
                            - omega_prime*np.arctan(tau_max*omega_prime)
            den_out_val = omega**2-omega_prime**2
            out_val = num_out_val/den_out_val
    elif block == 'im':
        # Kernel-function calculation
        if omega == omega_prime:
            out_val = 0.5*(-tau_max/(1+(tau_max*omega)**2)\
                            + np.arctan(tau_max*omega)/omega)
        else:
            num_out_val = omega*np.arctan(tau_max*omega_prime)\
                            - omega_prime*np.arctan(tau_max*omega)
            den_out_val = omega**2-omega_prime**2
            out_val = num_out_val/den_out_val
    elif block == 're-im':
        # Kernel-function calculation
        if omega == omega_prime:
            out_val = -tau_max**2*omega/(2.+2.*(tau_max*omega)**2)
        else:
            arg_log_num = 1+(tau_max*omega)**2
            arg_log_den = 1+(tau_max*omega_prime)**2
            num_out_val = -omega_prime*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val
    elif block == 'im-re':
        # Kernel-function calculation
        if omega == omega_prime:
            out_val = -tau_max**2*omega/(2.+2.*(tau_max*omega)**2)
        else:
            arg_log_num = 1+(tau_max*omega)**2
            arg_log_den = 1+(tau_max*omega_prime)**2
            num_out_val = -omega*(np.log(arg_log_num)-np.log(arg_log_den))
            den_out_val = 2*(omega**2-omega_prime**2)
            out_val = num_out_val/den_out_val
    else:
        out_val = 0.0
    out_val = (sigma_DRT**2)*out_val
    return out_val
# Kernel-function calculation
# Kernel-function calculation
# Kernel-function calculation
def k_0(x, ker_opts):
    # Parameter and hyperparameter settings
    sigma_SB = ker_opts['sigma_SB']
    ell = ker_opts['ell']  # Scale the input magnitude to improve numerical conditioning
    SB_ker_type = ker_opts['SB_ker_type'] # Kernel-function calculation
    a = 1./(sqrt(2)*ell)*x
    # Kernel-function calculation
    if SB_ker_type == 'IQ':
        out_val = 1/(1.+a**2)
    # Kernel-function calculation
    elif SB_ker_type == 'SE':
        out_val = np.exp(-a**2)
    out_val = (sigma_SB**2)*out_val
    return out_val
# Kernel-function calculation
def k_0_H(x, ker_opts):
    # Parameter and hyperparameter settings
    sigma_SB = ker_opts['sigma_SB']
    ell = ker_opts['ell']  # Scale the input magnitude to improve numerical conditioning
    SB_ker_type = ker_opts['SB_ker_type'] # Kernel-function calculation
    a = 1./(sqrt(2)*ell)*x
    # Kernel-function calculation
    if SB_ker_type == 'IQ':
        out_val = -a/(1.+a**2)
    # Kernel-function calculation
    elif SB_ker_type == 'SE':
        out_val = 2./sqrt(pi)*dawsn(a)
    out_val = (sigma_SB**2)*out_val
    return out_val
# Kernel-function calculation
def k_SB(omega, omega_prime, ker_opts, block):
    if block == 're':
        out_val = k_0(omega-omega_prime, ker_opts) + k_0(omega+omega_prime, ker_opts)
    elif block == 'im':
        out_val = k_0(omega-omega_prime, ker_opts) - k_0(omega+omega_prime, ker_opts)
    elif block == 're-im':
        out_val = -k_0_H(omega_prime-omega, ker_opts) - k_0_H(omega+omega_prime, ker_opts)
    elif block == 'im-re':
        out_val = -k_0_H(omega-omega_prime, ker_opts) - k_0_H(omega+omega_prime, ker_opts)
    else:
        out_val = 0
    return out_val
# Matrix dimensions and matrix operations
def mat_K(omega_m_vec, omega_n_vec, ker_opts, block):
    DRT_switch = ker_opts['DRT'] # Kernel-function calculation
    SB_switch = ker_opts['SB'] # Kernel-function calculation
    sigma_DRT = ker_opts['sigma_DRT']
    tau_max = ker_opts['tau_max']
    # Matrix dimensions and matrix operations
    N_m_freqs = omega_m_vec.size
    N_n_freqs = omega_n_vec.size
    K_mat = np.zeros([N_m_freqs, N_n_freqs])
    for m in range(0, N_m_freqs):
        for n in range(0, N_n_freqs):
            K_loc = 0.0
            # Kernel-function calculation
            if DRT_switch:
                k_DRT_loc = k_DRT(omega_m_vec[m], omega_n_vec[n], sigma_DRT, tau_max, block)
                K_loc += k_DRT_loc
            # Kernel-function calculation
            if SB_switch:
                k_SB_loc = k_SB(omega_m_vec[m], omega_n_vec[n], ker_opts, block)
                K_loc += k_SB_loc
            K_mat[m, n] = K_loc
    return K_mat
def NMLL_fct(theta, u, omega_vec, ker_opts_in, type_data):
    sigma_n = theta[0]
    sigma_DRT = theta[1]
    sigma_SB = theta[2]
    ell = theta[3]
    # Parameter and hyperparameter settings
    ker_opts = ker_opts_in.copy()
    ker_opts['sigma_SB'] = sigma_SB  # Kernel-function calculation
    ker_opts['sigma_DRT'] = sigma_DRT
    ker_opts['ell'] = ell  # Scale the input magnitude to improve numerical conditioning
    # Model settings
    N_freqs = omega_vec.size
    if type_data == 'im':
        sigma_L = theta[4]
        K_im = mat_K(omega_vec, omega_vec, ker_opts, type_data)
        Sigma = (sigma_n**2)*np.eye(N_freqs)
        K_full = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)
    elif type_data == 're':
        sigma_R = theta[4]
        K_re = mat_K(omega_vec, omega_vec, ker_opts, type_data)
        Sigma = (sigma_n**2)*np.eye(N_freqs)
        K_full = K_re + Sigma + (sigma_R**2)*np.ones_like(Sigma)
    else:
        sigma_R = theta[4]
        sigma_L = theta[5]
        K_full = np.zeros((2*N_freqs, 2*N_freqs))
        Sigma = (sigma_n**2)*np.eye(N_freqs)
        K_re = mat_K(omega_vec, omega_vec, ker_opts, 're')
        K_re_im = mat_K(omega_vec, omega_vec, ker_opts, 're-im')
        K_im_re = mat_K(omega_vec, omega_vec, ker_opts, 'im-re')
        K_im = mat_K(omega_vec, omega_vec, ker_opts, 'im')
        K_full[:N_freqs, :N_freqs] = K_re + Sigma + (sigma_R**2)*np.ones_like(Sigma)
        K_full[:N_freqs, N_freqs:] = K_re_im
        K_full[N_freqs:, :N_freqs] = K_im_re
        K_full[N_freqs:, N_freqs:] = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)
    # Cholesky decomposition for Gaussian-process linear algebra
    if not is_PD(K_full):
        K_full = nearest_PD(K_full)
    # Cholesky decomposition for Gaussian-process linear algebra
    L = np.linalg.cholesky(K_full)
    alpha = np.linalg.solve(L, u)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(u, alpha) + np.sum(np.log(np.diag(L)))
def compute_mu_sigma(theta, ker_opts, inv_K_full, freq_virt_vec, freq_vec, Z_exp_all):
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    # Frequency range and number of points
    N_freqs = freq_vec.size
    # Frequency range and number of points
    N_virt_freqs = freq_virt_vec.size
    # Frequency range and number of points
    omega_vec = 2.*pi*freq_vec
    omega_virt_vec = 2.*pi*freq_virt_vec
    mu_re_virt_vec = np.zeros_like(omega_virt_vec)
    sigma_re_virt_vec = np.zeros_like(omega_virt_vec)
    mu_im_virt_vec = np.zeros_like(omega_virt_vec)
    sigma_im_virt_vec = np.zeros_like(omega_virt_vec)
    for index, omega_virt in enumerate(omega_virt_vec):
        # Frequency range and number of points
        omega_virt_np = np.array([omega_virt])
        k_virt_virt_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're').flatten() + (sigma_R**2)
        k_virt_virt_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_virt_np**2
        k_virt_re_re = mat_K(omega_virt_np, omega_vec, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs)
        k_virt_re_im = mat_K(omega_virt_np, omega_vec, ker_opts, 're-im').flatten()
        k_virt_im_re = mat_K(omega_virt_np, omega_vec, ker_opts, 'im-re').flatten()
        k_virt_im_im = mat_K(omega_virt_np, omega_vec, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec*omega_virt_np
        k_virt_re = np.zeros(2*N_freqs)
        k_virt_re[:N_freqs] = k_virt_re_re
        k_virt_re[N_freqs:] = k_virt_re_im
        k_virt_im = np.zeros(2*N_freqs)
        k_virt_im[:N_freqs] = k_virt_im_re
        k_virt_im[N_freqs:] = k_virt_im_im
        mu_re_virt_vec[index] = k_virt_re@(inv_K_full@Z_exp_all)
        sigma_re_virt_vec[index] = np.sqrt(k_virt_virt_re - k_virt_re@(inv_K_full@k_virt_re))
        mu_im_virt_vec[index] = k_virt_im@(inv_K_full@Z_exp_all)
        sigma_im_virt_vec[index] = np.sqrt(k_virt_virt_im - k_virt_im@(inv_K_full@k_virt_im))
    return mu_re_virt_vec, mu_im_virt_vec, sigma_re_virt_vec, sigma_im_virt_vec
def compute_K_inv(theta, ker_opts, freq_vec):
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    # Frequency range and number of points
    N_freqs = freq_vec.size
    # Frequency range and number of points
    omega_vec = 2.*pi*freq_vec
    K_re = mat_K(omega_vec, omega_vec, ker_opts, 're')
    K_im = mat_K(omega_vec, omega_vec, ker_opts, 'im')
    K_re_im = mat_K(omega_vec, omega_vec, ker_opts, 're-im')
    K_im_re = mat_K(omega_vec, omega_vec, ker_opts, 'im-re')
    # Noise-condition settings
    Sigma = sigma_n**2*np.eye(N_freqs)
    # Matrix dimensions and matrix operations
    K_full = np.zeros((2*N_freqs, 2*N_freqs))
    K_full[:N_freqs, :N_freqs] = K_re + Sigma + (sigma_R**2)*np.ones(N_freqs)
    K_full[:N_freqs, N_freqs:] = K_re_im
    K_full[N_freqs:, :N_freqs] = K_im_re
    K_full[N_freqs:, N_freqs:] = K_im + Sigma + (sigma_L**2)*np.outer(omega_vec, omega_vec)
    if not is_PD(K_full):
        K_full = nearest_PD(K_full)
    # Cholesky decomposition for Gaussian-process linear algebra
    L = np.linalg.cholesky(K_full)
    # Matrix dimensions and matrix operations
    inv_L = np.linalg.inv(L)
    inv_K_full = np.dot(inv_L.T, inv_L)
    return inv_K_full
def compute_ALM(log10_freq_virt, theta, ker_opts, freq_vec, inv_K_full):
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    N_freqs = freq_vec.size
    omega_vec = 2.*pi*freq_vec
    omega_virt = 2.*pi*(10**log10_freq_virt)
    omega_virt_np = np.array([omega_virt])
    k_virt_virt_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're').flatten() + (sigma_R**2)
    k_virt_virt_re_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 're-im').flatten()
    k_virt_virt_im_re = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im-re').flatten()
    k_virt_virt_im = mat_K(omega_virt_np, omega_virt_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_virt_np**2
    k_virt_virt = np.zeros((2, 2))
    k_virt_virt[0, 0] = k_virt_virt_re
    k_virt_virt[0, 1] = k_virt_virt_re_im
    k_virt_virt[1, 0] = k_virt_virt_im_re
    k_virt_virt[1, 1] = k_virt_virt_im
    k_virt_re_re = mat_K(omega_virt_np, omega_vec, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs)
    k_virt_re_im = mat_K(omega_virt_np, omega_vec, ker_opts, 're-im').flatten()
    k_virt_im_re = mat_K(omega_virt_np, omega_vec, ker_opts, 'im-re').flatten()
    k_virt_im_im = mat_K(omega_virt_np, omega_vec, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec*omega_virt_np
    k_virt = np.zeros((2, 2*N_freqs))
    k_virt[0, :N_freqs] = k_virt_re_re
    k_virt[0, N_freqs:] = k_virt_re_im
    k_virt[1, :N_freqs] = k_virt_im_re
    k_virt[1, N_freqs:] = k_virt_im_im
    covariance = k_virt_virt - k_virt@(inv_K_full@k_virt.T)
    indicator = np.linalg.det(covariance)
    # Posterior mean and covariance calculation
    # Posterior mean and covariance calculation
    sigma_tot = -indicator
    return sigma_tot
def compute_ALC(log10_freq_Np1, theta, ker_opts, freq_vec_N, log10_freq_vec_int):
    sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    # Frequency range and number of points
    freq_Np1 = 10**log10_freq_Np1
    index = np.searchsorted(freq_vec_N, freq_Np1)
    # Frequency range and number of points
    freq_vec_Np1 = np.insert(freq_vec_N, index, freq_Np1)
    omega_vec_Np1 = 2.0*pi*freq_vec_Np1
    # Frequency range and number of points
    N_freqs_Np1 = freq_vec_Np1.size
    # Frequency range and number of points
    inv_K_Np1 = compute_K_inv(theta, ker_opts, freq_vec_Np1)
    indicator = np.zeros_like(log10_freq_vec_int)
    for iter, log10_freq_int in enumerate(log10_freq_vec_int):
        omega_int = 2.0*pi*(10**log10_freq_int)
        omega_int_np = np.array([omega_int])
        k_int_int_re = mat_K(omega_int_np, omega_int_np, ker_opts, 're').flatten() + (sigma_R**2)
        k_int_int_re_im = mat_K(omega_int_np, omega_int_np, ker_opts, 're-im').flatten()
        k_int_int_im_re = mat_K(omega_int_np, omega_int_np, ker_opts, 'im-re').flatten()
        k_int_int_im = mat_K(omega_int_np, omega_int_np, ker_opts, 'im').flatten() + (sigma_L**2)*omega_int_np**2
        k_int_int = np.zeros((2, 2))
        k_int_int[0, 0] = k_int_int_re
        k_int_int[0, 1] = k_int_int_re_im
        k_int_int[1, 0] = k_int_int_im_re
        k_int_int[1, 1] = k_int_int_im
        k_int_re_re = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 're').flatten() + (sigma_R**2)*np.ones(N_freqs_Np1)
        k_int_re_im = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 're-im').flatten()
        k_int_im_re = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 'im-re').flatten()
        k_int_im_im = mat_K(omega_int_np, omega_vec_Np1, ker_opts, 'im').flatten() + (sigma_L**2)*omega_vec_Np1*omega_int_np
        k_int = np.zeros((2, 2*N_freqs_Np1))
        k_int[0, :N_freqs_Np1] = k_int_re_re
        k_int[0, N_freqs_Np1:] = k_int_re_im
        k_int[1, :N_freqs_Np1] = k_int_im_re
        k_int[1, N_freqs_Np1:] = k_int_im_im
        covariance = k_int_int - k_int@(inv_K_Np1@k_int.T)
        indicator[iter] = np.linalg.det(covariance)
        # Posterior mean and covariance calculation
        # Posterior mean and covariance calculation
    ALC_out = np.trapz(indicator, x=log10_freq_vec_int)
    return ALC_out
def compute_opt_theta(theta_0, ker_opts_0, freq_vec, Z_exp_all, type_data = 'all'):
    omega_vec = 2.*pi*freq_vec
    if type_data == 'all':
        def print_results(theta):
            print('%.4E, %.4E, %.4E, %.4E, %.4E, %.6E; evidence = %.8E'%(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], NMLL_fct(theta, Z_exp_all, omega_vec, ker_opts_0, type_data)))
    else:
        def print_results(theta):
            print('%.4E, %.4E, %.4E, %.4E, %.6E; evidence = %.8E'%(theta[0], theta[1], theta[2], theta[3], theta[4], NMLL_fct(theta, Z_exp_all, omega_vec, ker_opts_0, type_data)))
    res = minimize(NMLL_fct, theta_0, args=(Z_exp_all, omega_vec, ker_opts_0, type_data), method='Powell', \
                        callback=print_results, options={'disp': True, 'xtol': 1E-6, 'ftol': 1E-6})
    theta = res.x
    if type_data == 'all':
        sigma_n, sigma_DRT, sigma_SB, ell, sigma_R, sigma_L = theta
    else:
        sigma_n, sigma_DRT, sigma_SB, ell, sigma_L = theta
    ker_opts = ker_opts_0.copy()
    ker_opts['sigma_SB'] = sigma_SB
    ker_opts['ell'] = ell
    ker_opts['sigma_DRT'] = sigma_DRT
    return theta, ker_opts
def update_exp(freq_new, Z_exp_new, freq_vec_N, Z_exp_Np):
    index = np.searchsorted(freq_vec_N, freq_new)
    # Frequency range and number of points
    freq_vec_Np1 = np.insert(freq_vec_N, index, freq_new)
    N_freqs_Np1 = freq_vec_Np1.size
    Z_exp_Np1 = np.insert(Z_exp_Np, index, Z_exp_new)
    Z_exp_all_Np1 = np.zeros(2*N_freqs_Np1)
    Z_exp_all_Np1[:N_freqs_Np1] = Z_exp_Np1.real
    Z_exp_all_Np1[N_freqs_Np1:] = Z_exp_Np1.imag
    return freq_vec_Np1, Z_exp_Np1, Z_exp_all_Np1
def res_score(res, band):
    count = np.zeros(3)
    for k in range(3):
        count[k] = np.sum(np.logical_and(res < (k+1)*band, res > -(k+1)*band))
    return count/len(res)
