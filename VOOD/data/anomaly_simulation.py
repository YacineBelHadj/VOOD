import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def simulate_anomaly(Pxx_den:np.ndarray,
                    f:np.ndarray, 
                    mode_of_interest:float,
                    shift:float, 
                    window:float):
    """Simulate an anomaly by shifting the mode of interest to a new mode$
    Pxx_den: Power spectral density of the signal: np.ndarray of 1 dim
    f: Frequency of the signal: np.ndarray of 1 dim
    mode_of_interest: Mode of interest: float
    shift: Shift of the mode of interest: float
    window: Window of the translation: int
    """
    new_mode = mode_of_interest + shift
    window = int(window / (f[1] - f[0]) / 2)
    arg_NM = np.argmin(np.abs(f - new_mode))
    arg_MOI = np.argmin(np.abs(f - mode_of_interest))
    Pxx_den_a = Pxx_den.copy()
    empty_region = Pxx_den_a[arg_MOI - window : arg_MOI + window]
    x = np.arange(len(empty_region))
    f_interp = interp1d(x, empty_region, kind='cubic')
    interpolated_values = f_interp(np.linspace(0, len(empty_region)-1, window*2))
    Pxx_den_a[arg_MOI - window : arg_MOI + window] = interpolated_values
    Pxx_den_a[arg_NM - window : arg_NM + window] = Pxx_den[arg_MOI - window : arg_MOI + window]
    return Pxx_den_a
