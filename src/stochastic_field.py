import numpy as np
import pandas as pd
import scipy.stats as stats

class StochasticFieldModel:
    def __init__(self, data_path, underlying, expiry):
        self.data_path = data_path
        self.underlying = underlying
        self.expiry = expiry
        self.chain_data = None 

    def calculate_gamma(self, S, K, T, r=0.05, sigma=0.15):
        """
        Calculates Black-Scholes Gamma.
        """
        T = np.maximum(T, 1e-5) # Prevent division by zero at expiry
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma

    def compute_potential_field(self, spot_array, current_chain, T):
        """
        Computes the aggregate gravitational potential field U(S).
        U(S) = -Sum [ OI(K) * Gamma(S,K,T) * (S-K)^2 ]
        """
        strikes = current_chain['Strike'].values
        oi = current_chain['Open Interest'].values
        
        # Reshape for broadcasting
        S_matrix = spot_array[:, np.newaxis]
        K_matrix = strikes[np.newaxis, :]
        OI_matrix = oi[np.newaxis, :]
        
        # Vectorized calculation
        Gamma_matrix = self.calculate_gamma(S_matrix, K_matrix, T)
        Distance_sq_matrix = (S_matrix - K_matrix)**2
        Components_matrix = OI_matrix * Gamma_matrix * Distance_sq_matrix
        
        # Sum across strikes and apply negative sign
        U_S = -np.sum(Components_matrix, axis=1)
        return U_S

    def compute_force_vector(self, spot_array, U_S):
        """
        Calculates the deterministic drift force.
        F(S) = -dU/dS
        """
        dS = spot_array[1] - spot_array[0]
        dU_dS = np.gradient(U_S, dS)
        force = -dU_dS
        return force

    def find_attractor(self, spot_array, U_S):
        """
        Finds the theoretical spot price S* where the field potential is minimized.
        """
        min_index = np.argmin(U_S)
        S_star = spot_array[min_index]
        return S_star