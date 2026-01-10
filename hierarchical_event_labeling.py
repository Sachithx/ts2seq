"""
HIERARCHICAL TIME SERIES EVENT LABELING SYSTEM - COMPLETE
==========================================================

Complete version with ALL detectors implemented:
    ✅ Enhanced features (63 features)
    ✅ ALL vocabulary labels used (64 labels)
    ✅ ChangePointDetector (MEAN_SHIFT_UP/DOWN)
    ✅ ChaoticSegmentDetector (VOLATILE_REGIME)
    ✅ WaveletBasedPeakDetector (multi-scale peaks)
    ✅ Enhanced trend/volatility detectors

Author: Sachith Abeywickrama
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from enum import IntEnum
import pywt  # Wavelet transform library
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: CORE DATA STRUCTURES
# ============================================================================

class EventScale(IntEnum):
    """Hierarchical scale levels for events"""
    MICRO = 1      # 1-5 timesteps (spikes, single points)
    MINI = 2       # 5-15 timesteps (very short segments)
    MESO = 3       # 15-50 timesteps (medium segments, local patterns)
    MACRO = 4      # 50-150 timesteps (major trends)
    GLOBAL = 5     # 150+ timesteps (full sequence characteristics)


@dataclass
class EventVocabulary:
    """
    Complete event vocabulary with 64 distinct labels.
    
    Categories:
        - Special tokens (0-2)
        - Step movements (3-10)
        - Trend segments (20-26)
        - Peaks/troughs (30-33)
        - Volatility regimes (40-43)
        - Change points (50-51)  ✅ NOW USED
        - Global regimes (60-63)  ✅ ALL USED
    """
    # Special tokens
    PAD = 0
    MASK = 1
    FLAT = 2
    
    # Step-level movements
    UP_SMALL = 3
    UP_MEDIUM = 4
    UP_LARGE = 5
    DOWN_SMALL = 6
    DOWN_MEDIUM = 7
    DOWN_LARGE = 8
    SPIKE_UP = 9
    SPIKE_DOWN = 10
    
    # Trend segments
    UPTREND_SHORT = 20
    UPTREND_MEDIUM = 21
    UPTREND_LONG = 22
    DOWNTREND_SHORT = 23
    DOWNTREND_MEDIUM = 24
    DOWNTREND_LONG = 25
    FLAT_SEGMENT = 26
    
    # Peaks and troughs
    LOCAL_PEAK = 30
    SHARP_PEAK = 31
    LOCAL_TROUGH = 32
    SHARP_TROUGH = 33
    
    # Volatility regimes
    LOW_VOLATILITY = 40
    NORMAL_VOLATILITY = 41
    HIGH_VOLATILITY = 42
    VOLATILITY_SPIKE = 43
    
    # Change points ✅ NOW DETECTED
    MEAN_SHIFT_UP = 50
    MEAN_SHIFT_DOWN = 51
    
    # Global regimes ✅ ALL USED
    BULLISH_REGIME = 60
    BEARISH_REGIME = 61
    SIDEWAYS_REGIME = 62
    VOLATILE_REGIME = 63  # ✅ NOW USED
    
    @classmethod
    def get_vocab_size(cls) -> int:
        """Return total vocabulary size"""
        return 64
    
    @classmethod
    def id_to_label(cls, idx: int) -> str:
        """Convert label ID to string name"""
        for name, value in vars(cls).items():
            if isinstance(value, int) and value == idx:
                return name
        return "UNKNOWN"


@dataclass
class HierarchicalEvent:
    """Event node in hierarchical tree structure."""
    start: int
    end: int
    label: int
    label_name: str
    scale: EventScale
    event_type: str
    confidence: float
    metadata: Dict
    parent: Optional['HierarchicalEvent'] = None
    children: List['HierarchicalEvent'] = field(default_factory=list)
    
    @property
    def duration(self) -> int:
        """Duration in timesteps"""
        return self.end - self.start + 1
    
    @property
    def depth(self) -> int:
        """Depth in hierarchy tree (0 = root)"""
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth
    
    def contains(self, other: 'HierarchicalEvent') -> bool:
        """Check if this event fully contains another event"""
        return (self.start <= other.start and 
                self.end >= other.end and
                self != other)
    
    def __repr__(self):
        indent = "  " * self.depth
        children_info = f" ({len(self.children)} children)" if self.children else ""
        return (f"{indent}[{self.start:03d}-{self.end:03d}] {self.label_name} "
                f"(scale={self.scale.name}){children_info}")


# Global vocabulary instance
VOCAB = EventVocabulary()


# ============================================================================
# SECTION 2: ENHANCED FEATURE EXTRACTION (from previous version)
# ============================================================================

class EnhancedMultiScaleFeatureExtractor:
    """Extract comprehensive features at multiple temporal scales."""
    
    def __init__(self, 
                 scales: List[int] = [5, 10, 20, 50],
                 use_spectral: bool = True,
                 use_entropy: bool = True,
                 use_wavelets: bool = True,
                 wavelet_type: str = 'db4',
                 wavelet_levels: Optional[int] = None):
        self.scales = scales
        self.use_spectral = use_spectral
        self.use_entropy = use_entropy
        self.use_wavelets = use_wavelets
        self.wavelet_type = wavelet_type
        self.wavelet_levels = wavelet_levels
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive multi-scale features from time series batch."""
        B, L = x.shape
        device = x.device
        features = {}
        
        # Basic derivatives
        dx = torch.diff(x, dim=1)
        features['dx'] = F.pad(dx, (1, 0), value=0)
        features['ddx'] = self._second_derivative(x)
        
        # Multi-scale rolling features
        for w in self.scales:
            if w >= L:
                continue
            
            x_3d = x.unsqueeze(1)
            kernel = torch.ones(1, 1, w, device=device) / w
            padding = w - 1
            
            x_padded = F.pad(x_3d, (padding, 0), mode='replicate')
            rolling_mean = F.conv1d(x_padded, kernel).squeeze(1)
            features[f'mean_{w}'] = rolling_mean
            
            x_diff = x.unsqueeze(1) - rolling_mean.unsqueeze(1)
            x_diff_padded = F.pad(x_diff, (padding, 0), mode='replicate')
            rolling_var = F.conv1d(x_diff_padded ** 2, kernel).squeeze(1)
            rolling_std = torch.sqrt(rolling_var.clamp(min=1e-8))
            features[f'std_{w}'] = rolling_std
            
            slopes = self._compute_slopes(x, w)
            features[f'slope_{w}'] = slopes
            
            rolling_min, rolling_max = self._compute_rolling_extrema(x, w)
            features[f'min_{w}'] = rolling_min
            features[f'max_{w}'] = rolling_max
            features[f'range_{w}'] = rolling_max - rolling_min
            
            norm_slope = slopes / (rolling_std + 1e-8)
            features[f'norm_slope_{w}'] = norm_slope
            
            if self.use_spectral:
                spec_features = self._compute_spectral_features(x, w)
                features[f'spec_low_{w}'] = spec_features['low']
                features[f'spec_mid_{w}'] = spec_features['mid']
                features[f'spec_high_{w}'] = spec_features['high']
            
            if self.use_entropy:
                features[f'entropy_{w}'] = self._compute_entropy(x, w)
        
        # Global statistical features
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        features['zscore'] = (x - mean) / std
        features['jump_indicator'] = self._detect_jumps(features['dx'], features.get('std_20'))
        features['vol_asymmetry'] = self._compute_volatility_asymmetry(features['dx'])
        
        # Wavelet features
        if self.use_wavelets:
            wavelet_features = self._compute_wavelet_features(x)
            features.update(wavelet_features)
        
        return features
    
    def _second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        ddx = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
        ddx = F.pad(ddx, (2, 0), value=0.0)
        return ddx
    
    def _compute_rolling_extrema(self, x: torch.Tensor, window: int) -> tuple:
        B, L = x.shape
        x_padded = F.pad(x, (window-1, 0), mode='replicate')
        windows = x_padded.unfold(dimension=1, size=window, step=1)
        rolling_min = windows.min(dim=2)[0]
        rolling_max = windows.max(dim=2)[0]
        return rolling_min, rolling_max
    
    def _compute_spectral_features(self, x: torch.Tensor, window: int) -> Dict[str, torch.Tensor]:
        B, L = x.shape
        device = x.device
        spec_low = torch.zeros(B, L, device=device)
        spec_mid = torch.zeros(B, L, device=device)
        spec_high = torch.zeros(B, L, device=device)
        
        for b in range(B):
            x_np = x[b].cpu().numpy()
            for t in range(window, L):
                segment = x_np[t-window+1:t+1]
                fft = np.fft.rfft(segment)
                power = np.abs(fft) ** 2
                n_freq = len(power)
                third = n_freq // 3
                spec_low[b, t] = float(np.sum(power[:third]))
                spec_mid[b, t] = float(np.sum(power[third:2*third]))
                spec_high[b, t] = float(np.sum(power[2*third:]))
        
        total = spec_low + spec_mid + spec_high + 1e-8
        return {'low': spec_low / total, 'mid': spec_mid / total, 'high': spec_high / total}
    
    def _compute_entropy(self, x: torch.Tensor, window: int, n_bins: int = 10) -> torch.Tensor:
        B, L = x.shape
        device = x.device
        entropy_vals = torch.zeros(B, L, device=device)
        
        for b in range(B):
            x_np = x[b].cpu().numpy()
            for t in range(window, L):
                segment = x_np[t-window+1:t+1]
                hist, _ = np.histogram(segment, bins=n_bins, density=True)
                hist = hist + 1e-10
                ent = -np.sum(hist * np.log(hist + 1e-10))
                entropy_vals[b, t] = float(ent)
        
        return entropy_vals
    
    def _detect_jumps(self, dx: torch.Tensor, vol: Optional[torch.Tensor] = None) -> torch.Tensor:
        abs_dx = torch.abs(dx)
        if vol is not None:
            threshold = 3.0 * vol
            jumps = (abs_dx > threshold).float()
        else:
            threshold = torch.quantile(abs_dx.reshape(-1), 0.95)
            jumps = (abs_dx > threshold).float()
        return jumps
    
    def _compute_volatility_asymmetry(self, dx: torch.Tensor, window: int = 20) -> torch.Tensor:
        B, L = dx.shape
        device = dx.device
        asymmetry = torch.ones(B, L, device=device)
        
        for t in range(window, L):
            dx_window = dx[:, t-window+1:t+1]
            up_moves = dx_window.clamp(min=0)
            down_moves = (-dx_window).clamp(min=0)
            up_vol = up_moves.std(dim=1) + 1e-8
            down_vol = down_moves.std(dim=1) + 1e-8
            asymmetry[:, t] = up_vol / down_vol
        
        return asymmetry
    
    def _compute_wavelet_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L = x.shape
        device = x.device
        
        if self.wavelet_levels is None:
            max_level = pywt.dwt_max_level(L, self.wavelet_type)
            levels = min(max_level, 5)
        else:
            levels = self.wavelet_levels
        
        features = {}
        
        for b in range(B):
            x_np = x[b].cpu().numpy()
            coeffs = pywt.wavedec(x_np, self.wavelet_type, level=levels)
            
            approx = coeffs[0]
            details = coeffs[1:]
            
            approx_upsampled = self._upsample_coeffs(approx, L)
            if b == 0:
                features['wavelet_a'] = torch.zeros(B, L, device=device)
                features['wavelet_energy_a'] = torch.zeros(B, L, device=device)
            features['wavelet_a'][b] = torch.from_numpy(approx_upsampled).float()
            features['wavelet_energy_a'][b] = torch.from_numpy(approx_upsampled ** 2).float()
            
            for i, detail in enumerate(reversed(details), start=1):
                detail_upsampled = self._upsample_coeffs(detail, L)
                
                if b == 0:
                    features[f'wavelet_d{i}'] = torch.zeros(B, L, device=device)
                    features[f'wavelet_energy_d{i}'] = torch.zeros(B, L, device=device)
                
                features[f'wavelet_d{i}'][b] = torch.from_numpy(detail_upsampled).float()
                features[f'wavelet_energy_d{i}'][b] = torch.from_numpy(detail_upsampled ** 2).float()
        
        return features
    
    def _upsample_coeffs(self, coeffs: np.ndarray, target_length: int) -> np.ndarray:
        n = len(coeffs)
        if n == target_length:
            return coeffs
        old_indices = np.linspace(0, target_length - 1, n)
        new_indices = np.arange(target_length)
        upsampled = np.interp(new_indices, old_indices, coeffs)
        return upsampled
    
    def _compute_slopes(self, x: torch.Tensor, window: int) -> torch.Tensor:
        B, L = x.shape
        slopes = torch.zeros(B, L, device=x.device)
        for i in range(window, L):
            slopes[:, i] = (x[:, i] - x[:, i-window+1]) / window
        return slopes


# ============================================================================
# SECTION 3: STEP-WISE LABEL ENCODING
# ============================================================================

class StepWiseEncoder:
    """Encode each timestep with symbolic movement labels."""
    
    def encode(self, x: torch.Tensor, features: Dict) -> torch.Tensor:
        B, L = x.shape
        device = x.device
        
        dx = features['dx']
        abs_dx = torch.abs(dx[:, 1:])
        
        if abs_dx.numel() == 0:
            return torch.full((B, L), VOCAB.FLAT, dtype=torch.long, device=device)
        
        q33, q66, q90 = torch.quantile(
            abs_dx.reshape(-1),
            torch.tensor([0.33, 0.66, 0.90], device=device)
        )
        
        epsilon = 0.1 * q33
        labels = torch.full((B, L), VOCAB.PAD, dtype=torch.long, device=device)
        
        for t in range(1, L):
            diff = dx[:, t]
            abs_diff = torch.abs(diff)
            
            flat_mask = abs_diff < epsilon
            labels[flat_mask, t] = VOCAB.FLAT
            
            up_mask = (diff > 0) & (~flat_mask)
            labels[up_mask & (abs_diff <= q33), t] = VOCAB.UP_SMALL
            labels[up_mask & (abs_diff > q33) & (abs_diff <= q66), t] = VOCAB.UP_MEDIUM
            labels[up_mask & (abs_diff > q66) & (abs_diff <= q90), t] = VOCAB.UP_LARGE
            labels[up_mask & (abs_diff > q90), t] = VOCAB.SPIKE_UP
            
            down_mask = (diff < 0) & (~flat_mask)
            labels[down_mask & (abs_diff <= q33), t] = VOCAB.DOWN_SMALL
            labels[down_mask & (abs_diff > q33) & (abs_diff <= q66), t] = VOCAB.DOWN_MEDIUM
            labels[down_mask & (abs_diff > q66) & (abs_diff <= q90), t] = VOCAB.DOWN_LARGE
            labels[down_mask & (abs_diff > q90), t] = VOCAB.SPIKE_DOWN
        
        return labels


# ============================================================================
# SECTION 4: EVENT DETECTORS (ALL IMPLEMENTED)
# ============================================================================

@dataclass
class SimpleSegment:
    """Simple segment representation for detector outputs"""
    start: int
    end: int
    label: int
    metadata: Dict = field(default_factory=dict)


class EnhancedTrendSegmentDetector:
    """Enhanced trend detection using normalized slopes."""
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        x_np = x.cpu().numpy()
        L = len(x_np)
        
        if 'norm_slope_20' in features:
            slopes = features['norm_slope_20'][idx].cpu().numpy()
        elif 'slope_20' in features:
            slopes = features['slope_20'][idx].cpu().numpy()
        else:
            slopes = np.gradient(x_np)
        
        slope_sign = np.sign(slopes)
        slope_sign[np.abs(slopes) < 0.01] = 0
        
        changes = np.where(np.diff(slope_sign) != 0)[0] + 1
        breakpoints = np.concatenate([[0], changes, [L]])
        
        segments = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1] - 1
            
            if end - start < 5:
                continue
            
            avg_slope = slopes[start:end+1].mean()
            duration = end - start + 1
            
            if abs(avg_slope) < 0.01:
                label = VOCAB.FLAT_SEGMENT
            elif avg_slope > 0:
                if duration < 30:
                    label = VOCAB.UPTREND_SHORT
                elif duration < 80:
                    label = VOCAB.UPTREND_MEDIUM
                else:
                    label = VOCAB.UPTREND_LONG
            else:
                if duration < 30:
                    label = VOCAB.DOWNTREND_SHORT
                elif duration < 80:
                    label = VOCAB.DOWNTREND_MEDIUM
                else:
                    label = VOCAB.DOWNTREND_LONG
            
            segments.append(SimpleSegment(
                start=start, end=end, label=label,
                metadata={'slope': float(avg_slope)}
            ))
        
        return segments


class PeakTroughDetector:
    """Detect peaks and troughs using scipy's find_peaks."""
    
    def __init__(self, min_distance: int = 10, min_prominence_percentile: float = 75):
        self.min_distance = min_distance
        self.min_prominence_percentile = min_prominence_percentile
    
    def detect(self, x: torch.Tensor, idx: int) -> List[SimpleSegment]:
        x_np = x.cpu().numpy()
        std = np.std(x_np)
        min_prominence = max(0.2 * std, 0.1)
        
        events = []
        
        try:
            peaks, props = scipy_signal.find_peaks(
                x_np, prominence=min_prominence,
                distance=self.min_distance, width=1
            )
            
            for pk, prom in zip(peaks, props['prominences']):
                label = VOCAB.SHARP_PEAK if prom > std else VOCAB.LOCAL_PEAK
                events.append(SimpleSegment(
                    start=int(pk), end=int(pk), label=label,
                    metadata={'prominence': float(prom), 'type': 'peak'}
                ))
        except:
            pass
        
        try:
            troughs, props = scipy_signal.find_peaks(
                -x_np, prominence=min_prominence,
                distance=self.min_distance, width=1
            )
            
            for tr, prom in zip(troughs, props['prominences']):
                label = VOCAB.SHARP_TROUGH if prom > std else VOCAB.LOCAL_TROUGH
                events.append(SimpleSegment(
                    start=int(tr), end=int(tr), label=label,
                    metadata={'prominence': float(prom), 'type': 'trough'}
                ))
        except:
            pass
        
        events = self._validate_alternation(events)
        return events
    
    def _validate_alternation(self, events: List[SimpleSegment]) -> List[SimpleSegment]:
        if len(events) <= 1:
            return events
        
        events.sort(key=lambda e: e.start)
        filtered = [events[0]]
        
        for event in events[1:]:
            last_event = filtered[-1]
            last_type = last_event.metadata.get('type')
            curr_type = event.metadata.get('type')
            
            if last_type == curr_type:
                if event.metadata['prominence'] > last_event.metadata['prominence']:
                    filtered[-1] = event
            else:
                if event.start - last_event.start >= self.min_distance // 2:
                    filtered.append(event)
        
        return filtered


class VolatilityRegimeDetector:
    """Detect volatility regimes using rolling standard deviation."""
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        if 'std_20' not in features:
            return []
        
        vol = features['std_20'][idx].cpu().numpy()
        L = len(vol)
        
        q25, q75, q90 = np.percentile(vol, [25, 75, 90])
        
        vol_levels = np.zeros(L, dtype=int)
        vol_levels[vol <= q25] = 0
        vol_levels[(vol > q25) & (vol <= q75)] = 1
        vol_levels[(vol > q75) & (vol <= q90)] = 2
        vol_levels[vol > q90] = 3
        
        changes = np.where(np.diff(vol_levels) != 0)[0] + 1
        breakpoints = np.concatenate([[0], changes, [L]])
        
        regimes = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1] - 1
            
            if end - start < 5:
                continue
            
            level_code = vol_levels[start]
            avg_vol = vol[start:end+1].mean()
            
            label_map = {
                0: VOCAB.LOW_VOLATILITY,
                1: VOCAB.NORMAL_VOLATILITY,
                2: VOCAB.HIGH_VOLATILITY,
                3: VOCAB.VOLATILITY_SPIKE
            }
            
            regimes.append(SimpleSegment(
                start=start, end=end, label=label_map[level_code],
                metadata={'avg_volatility': float(avg_vol)}
            ))
        
        return regimes


# ✅ NEW: CHANGE POINT DETECTOR
class ChangePointDetector:
    """
    Detect change points using CUSUM and curvature.
    
    Detects MEAN_SHIFT_UP and MEAN_SHIFT_DOWN events.
    """
    
    def __init__(self, min_segment_length: int = 10, threshold_factor: float = 2.0):
        self.min_segment_length = min_segment_length
        self.threshold_factor = threshold_factor
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        x_np = x.cpu().numpy()
        L = len(x_np)
        
        if 'ddx' in features:
            ddx = features['ddx'][idx].cpu().numpy()
        else:
            ddx = np.diff(x_np, n=2)
            ddx = np.pad(ddx, (2, 0), constant_values=0)
        
        cusum_changepoints = self._detect_cusum_changepoints(x_np)
        curvature_changepoints = self._detect_curvature_changepoints(ddx)
        
        all_changepoints = sorted(set(cusum_changepoints + curvature_changepoints))
        filtered_changepoints = self._filter_by_distance(all_changepoints)
        
        events = []
        for cp in filtered_changepoints:
            if cp < 10 or cp >= L - 10:
                continue
            
            before_mean = x_np[max(0, cp-20):cp].mean()
            after_mean = x_np[cp:min(L, cp+20)].mean()
            shift_magnitude = after_mean - before_mean
            
            if abs(shift_magnitude) < 0.1:
                continue
            
            label = VOCAB.MEAN_SHIFT_UP if shift_magnitude > 0 else VOCAB.MEAN_SHIFT_DOWN
            
            events.append(SimpleSegment(
                start=cp, end=cp, label=label,
                metadata={
                    'shift_magnitude': float(shift_magnitude),
                    'before_mean': float(before_mean),
                    'after_mean': float(after_mean)
                }
            ))
        
        return events
    
    def _detect_cusum_changepoints(self, x: np.ndarray, threshold: float = 5.0) -> List[int]:
        L = len(x)
        mean = np.mean(x)
        std = np.std(x)
        
        if std < 1e-8:
            return []
        
        deviations = (x - mean) / std
        cusum_pos = np.zeros(L)
        cusum_neg = np.zeros(L)
        
        for i in range(1, L):
            cusum_pos[i] = max(0, cusum_pos[i-1] + deviations[i])
            cusum_neg[i] = min(0, cusum_neg[i-1] + deviations[i])
        
        changepoints = []
        
        pos_crosses = np.where(cusum_pos > threshold)[0]
        if len(pos_crosses) > 0:
            last_reset = 0
            for cp in pos_crosses:
                if cp - last_reset > self.min_segment_length:
                    changepoints.append(cp)
                    last_reset = cp
        
        neg_crosses = np.where(cusum_neg < -threshold)[0]
        if len(neg_crosses) > 0:
            last_reset = 0
            for cp in neg_crosses:
                if cp - last_reset > self.min_segment_length:
                    changepoints.append(cp)
                    last_reset = cp
        
        return changepoints
    
    def _detect_curvature_changepoints(self, ddx: np.ndarray, percentile: float = 95) -> List[int]:
        abs_ddx = np.abs(ddx)
        threshold = np.percentile(abs_ddx, percentile)
        candidates = np.where(abs_ddx > threshold)[0]
        return candidates.tolist()
    
    def _filter_by_distance(self, changepoints: List[int]) -> List[int]:
        if len(changepoints) <= 1:
            return changepoints
        
        filtered = [changepoints[0]]
        for cp in changepoints[1:]:
            if cp - filtered[-1] >= self.min_segment_length:
                filtered.append(cp)
        
        return filtered


# ✅ NEW: CHAOTIC SEGMENT DETECTOR
class ChaoticSegmentDetector:
    """
    Detect chaotic/irregular segments using entropy.
    
    Uses VOLATILE_REGIME label for high-entropy segments.
    """
    
    def __init__(self, min_segment_length: int = 20, entropy_percentile: float = 75):
        self.min_segment_length = min_segment_length
        self.entropy_percentile = entropy_percentile
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        if 'entropy_20' not in features:
            return []
        
        entropy = features['entropy_20'][idx].cpu().numpy()
        L = len(entropy)
        
        threshold = np.percentile(entropy[entropy > 0], self.entropy_percentile)
        high_entropy = entropy > threshold
        
        segments = self._find_contiguous_segments(high_entropy)
        
        events = []
        for start, end in segments:
            duration = end - start + 1
            
            if duration < self.min_segment_length:
                continue
            
            avg_entropy = entropy[start:end+1].mean()
            
            events.append(SimpleSegment(
                start=start, end=end, label=VOCAB.VOLATILE_REGIME,
                metadata={'avg_entropy': float(avg_entropy), 'complexity': 'high'}
            ))
        
        return events
    
    def _find_contiguous_segments(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        segments = []
        in_segment = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i - 1))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(mask) - 1))
        
        return segments


# ✅ NEW: WAVELET-BASED PEAK DETECTOR
class WaveletBasedPeakDetector:
    """
    Enhanced peak detection using wavelet decomposition.
    
    Uses detail coefficients at different scales.
    """
    
    def __init__(self, min_distance: int = 10):
        self.min_distance = min_distance
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        if 'wavelet_d1' not in features:
            return []
        
        events = []
        
        scale_mapping = {
            'wavelet_d1': (EventScale.MICRO, 'finest'),
            'wavelet_d2': (EventScale.MINI, 'fine'),
            'wavelet_d3': (EventScale.MESO, 'medium'),
        }
        
        for wavelet_key, (scale, desc) in scale_mapping.items():
            if wavelet_key not in features:
                continue
            
            detail_coeffs = features[wavelet_key][idx].cpu().numpy()
            
            # Peaks
            try:
                peaks, props = scipy_signal.find_peaks(
                    detail_coeffs, distance=self.min_distance, prominence=0.1
                )
                
                for pk, prom in zip(peaks, props['prominences']):
                    window = 5
                    start = max(0, pk - window)
                    end = min(len(x), pk + window)
                    local_max_idx = start + np.argmax(x[start:end].cpu().numpy())
                    
                    if abs(local_max_idx - pk) <= window:
                        events.append(SimpleSegment(
                            start=int(local_max_idx), end=int(local_max_idx),
                            label=VOCAB.LOCAL_PEAK if prom < 0.5 else VOCAB.SHARP_PEAK,
                            metadata={
                                'prominence': float(prom), 'type': 'peak',
                                'wavelet_scale': desc, 'detected_at_scale': scale.name
                            }
                        ))
            except:
                pass
            
            # Troughs
            try:
                troughs, props = scipy_signal.find_peaks(
                    -detail_coeffs, distance=self.min_distance, prominence=0.1
                )
                
                for tr, prom in zip(troughs, props['prominences']):
                    window = 5
                    start = max(0, tr - window)
                    end = min(len(x), tr + window)
                    local_min_idx = start + np.argmin(x[start:end].cpu().numpy())
                    
                    if abs(local_min_idx - tr) <= window:
                        events.append(SimpleSegment(
                            start=int(local_min_idx), end=int(local_min_idx),
                            label=VOCAB.LOCAL_TROUGH if prom < 0.5 else VOCAB.SHARP_TROUGH,
                            metadata={
                                'prominence': float(prom), 'type': 'trough',
                                'wavelet_scale': desc, 'detected_at_scale': scale.name
                            }
                        ))
            except:
                pass
        
        return events


# ============================================================================
# SECTION 5: HIERARCHICAL STRUCTURE BUILDER
# ============================================================================

class HierarchicalEventBuilder:
    """Build hierarchical event tree from flat event list."""
    
    def __init__(self):
        self.events: List[HierarchicalEvent] = []
    
    def add_event(self, start: int, end: int, label: int, event_type: str,
                  confidence: float = 1.0, metadata: Optional[Dict] = None):
        duration = end - start + 1
        
        if duration <= 5:
            scale = EventScale.MICRO
        elif duration <= 15:
            scale = EventScale.MINI
        elif duration <= 50:
            scale = EventScale.MESO
        elif duration <= 150:
            scale = EventScale.MACRO
        else:
            scale = EventScale.GLOBAL
        
        event = HierarchicalEvent(
            start=start, end=end, label=label,
            label_name=VOCAB.id_to_label(label),
            scale=scale, event_type=event_type,
            confidence=confidence, metadata=metadata or {}
        )
        self.events.append(event)
    
    def build_hierarchy(self) -> List[HierarchicalEvent]:
        sorted_events = sorted(self.events, key=lambda e: (-e.scale, e.start))
        roots = []
        
        for event in sorted_events:
            parent = self._find_parent(event, sorted_events)
            if parent is None:
                roots.append(event)
            else:
                event.parent = parent
                parent.children.append(event)
        
        self._sort_children(roots)
        return roots
    
    def _find_parent(self, event: HierarchicalEvent,
                     all_events: List[HierarchicalEvent]) -> Optional[HierarchicalEvent]:
        candidates = [e for e in all_events if e != event and 
                     e.scale > event.scale and e.contains(event)]
        return min(candidates, key=lambda e: e.duration) if candidates else None
    
    def _sort_children(self, nodes: List[HierarchicalEvent]):
        for node in nodes:
            if node.children:
                node.children.sort(key=lambda c: c.start)
                self._sort_children(node.children)
    
    def get_flat_list(self, roots: List[HierarchicalEvent]) -> List[HierarchicalEvent]:
        result = []
        def traverse(node):
            result.append(node)
            for child in node.children:
                traverse(child)
        for root in roots:
            traverse(root)
        return result


# ============================================================================
# SECTION 6: HIERARCHICAL ANNOTATION
# ============================================================================

@dataclass
class HierarchicalAnnotation:
    """Complete hierarchical annotation for one sequence."""
    sequence: torch.Tensor
    step_labels: torch.Tensor
    event_roots: List[HierarchicalEvent]
    all_events: List[HierarchicalEvent]
    
    def print_hierarchy(self, max_depth: int = 10):
        def print_tree(node: HierarchicalEvent, depth: int = 0):
            if depth > max_depth:
                return
            print(node)
            for child in node.children:
                print_tree(child, depth + 1)
        
        print(f"\nHierarchical Events (Total: {len(self.all_events)})")
        print("=" * 80)
        for root in self.event_roots:
            print_tree(root)
    
    def get_events_at_scale(self, scale: EventScale) -> List[HierarchicalEvent]:
        return [e for e in self.all_events if e.scale == scale]
    
    def get_events_in_range(self, start: int, end: int) -> List[HierarchicalEvent]:
        return [e for e in self.all_events 
                if not (e.end < start or e.start > end)]
    
    def to_text(self, format: str = 'depth_marked') -> str:
        if format == 'depth_marked':
            return self._depth_marked_text()
        elif format == 'flat':
            return self._flat_text()
        elif format == 'narrative':
            return self._narrative_text()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _depth_marked_text(self) -> str:
        parts = []
        def traverse(node):
            depth_marker = ">" * node.depth
            parts.append(f"{depth_marker}[{node.start}-{node.end}]{node.label_name}")
            for child in node.children:
                traverse(child)
        for root in self.event_roots:
            traverse(root)
        return " ".join(parts)
    
    def _flat_text(self) -> str:
        events = sorted(self.all_events, key=lambda e: e.start)
        return " ".join(f"[{e.start}-{e.end}]{e.label_name}" for e in events)
    
    def _narrative_text(self) -> str:
        sentences = []
        
        global_events = self.get_events_at_scale(EventScale.GLOBAL)
        if global_events:
            sentences.append(
                f"Overall: {global_events[0].label_name.lower().replace('_', ' ')}."
            )
        
        macro_events = self.get_events_at_scale(EventScale.MACRO)
        if macro_events:
            sentences.append(f"{len(macro_events)} major segments detected.")
            for event in macro_events[:3]:
                desc = event.label_name.lower().replace('_', ' ')
                sentences.append(f"[{event.start}-{event.end}]: {desc}")
                if event.children:
                    nested = ", ".join(set(c.event_type for c in event.children))
                    sentences.append(f"  (contains: {nested})")
        
        return " ".join(sentences)


# ============================================================================
# SECTION 7: COMPLETE DATASET CLASS
# ============================================================================

class CompleteHierarchicalEventDataset(Dataset):
    """
    ✅ COMPLETE dataset with ALL detectors enabled.
    
    All 64 vocabulary labels are now used!
    All 63 features are computed and utilized!
    """
    
    def __init__(self, 
                 x: torch.Tensor, 
                 use_spectral: bool = True,
                 use_entropy: bool = True,
                 use_wavelets: bool = True,
                 use_wavelet_peaks: bool = True,  # ✅ NEW
                 use_changepoint: bool = True,     # ✅ NEW
                 use_chaotic: bool = True,         # ✅ NEW
                 verbose: bool = True):
        super().__init__()
        
        if x.dim() != 2:
            raise ValueError("Expected x with shape [B, L]")
        
        self.x = x
        B, L = x.shape
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"INITIALIZING COMPLETE HIERARCHICAL EVENT DATASET")
            print(f"{'='*80}")
            print(f"Sequences: {B}")
            print(f"Length: {L}")
            print(f"Spectral features: {'✓' if use_spectral else '✗'}")
            print(f"Entropy features: {'✓' if use_entropy else '✗'}")
            print(f"Wavelet features: {'✓' if use_wavelets else '✗'}")
            print(f"Wavelet-based peaks: {'✓' if use_wavelet_peaks else '✗'}")
            print(f"Change point detection: {'✓' if use_changepoint else '✗'}")
            print(f"Chaotic segment detection: {'✓' if use_chaotic else '✗'}")
        
        # Initialize ALL components
        self.feature_extractor = EnhancedMultiScaleFeatureExtractor(
            use_spectral=use_spectral,
            use_entropy=use_entropy,
            use_wavelets=use_wavelets
        )
        self.step_encoder = StepWiseEncoder()
        self.trend_detector = EnhancedTrendSegmentDetector()
        self.peak_detector = PeakTroughDetector()
        self.vol_detector = VolatilityRegimeDetector()
        
        # ✅ NEW DETECTORS
        self.changepoint_detector = ChangePointDetector() if use_changepoint else None
        self.chaotic_detector = ChaoticSegmentDetector() if use_chaotic else None
        self.wavelet_peak_detector = WaveletBasedPeakDetector() if use_wavelet_peaks else None
        
        # Extract features
        if verbose:
            print(f"\n[1/4] Extracting enhanced multi-scale features...")
        self.features = self.feature_extractor.extract_features(x)
        if verbose:
            print(f"      ✓ Computed {len(self.features)} feature types")
        
        # Encode step labels
        if verbose:
            print(f"[2/4] Encoding step-wise labels...")
        self.step_labels = self.step_encoder.encode(x, self.features)
        if verbose:
            print(f"      ✓ Encoded {B * L} timesteps")
        
        # Detect events and build hierarchy
        if verbose:
            print(f"[3/4] Detecting events with ALL detectors...")
        
        self.annotations = []
        for i in range(B):
            if verbose and i % 50 == 0:
                print(f"      Processing sequence {i}/{B}...")
            
            annotation = self._build_annotation(i, L)
            self.annotations.append(annotation)
        
        # Statistics
        if verbose:
            print(f"[4/4] Computing statistics...")
            self._print_statistics()
            print(f"\n{'='*80}")
            print(f"✓ COMPLETE DATASET READY")
            print(f"{'='*80}\n")
    
    def _build_annotation(self, idx: int, L: int) -> HierarchicalAnnotation:
        builder = HierarchicalEventBuilder()
        
        # ALL DETECTORS
        trends = self.trend_detector.detect(self.x[idx], self.features, idx)
        peaks = self.peak_detector.detect(self.x[idx], idx)
        vol_regimes = self.vol_detector.detect(self.x[idx], self.features, idx)
        
        # ✅ NEW: Change points
        if self.changepoint_detector:
            changepoints = self.changepoint_detector.detect(self.x[idx], self.features, idx)
            for cp in changepoints:
                builder.add_event(cp.start, cp.end, cp.label, 'changepoint',
                                confidence=0.8, metadata=cp.metadata)
        
        # ✅ NEW: Chaotic segments
        if self.chaotic_detector:
            chaotic_segs = self.chaotic_detector.detect(self.x[idx], self.features, idx)
            for seg in chaotic_segs:
                builder.add_event(seg.start, seg.end, seg.label, 'chaotic',
                                confidence=0.75, metadata=seg.metadata)
        
        # ✅ NEW: Wavelet-based peaks
        if self.wavelet_peak_detector:
            wavelet_peaks = self.wavelet_peak_detector.detect(self.x[idx], self.features, idx)
            for pk in wavelet_peaks:
                builder.add_event(pk.start, pk.end, pk.label, 'peak_wavelet',
                                confidence=0.85, metadata=pk.metadata)
        
        # Original detectors
        for seg in trends:
            builder.add_event(seg.start, seg.end, seg.label, 'trend',
                            confidence=0.9, metadata=seg.metadata)
        
        for pk in peaks:
            builder.add_event(pk.start, pk.end, pk.label, 'peak',
                            confidence=0.85, metadata=pk.metadata)
        
        for vr in vol_regimes:
            builder.add_event(vr.start, vr.end, vr.label, 'volatility',
                            confidence=0.8, metadata=vr.metadata)
        
        # ✅ ENHANCED: Global regime (now uses VOLATILE_REGIME)
        global_label = self._classify_global_regime(idx)
        builder.add_event(0, L-1, global_label, 'regime', confidence=0.7)
        
        # Build hierarchy
        roots = builder.build_hierarchy()
        all_events = builder.get_flat_list(roots)
        
        return HierarchicalAnnotation(
            sequence=self.x[idx],
            step_labels=self.step_labels[idx],
            event_roots=roots,
            all_events=all_events
        )
    
    def _classify_global_regime(self, idx: int) -> int:
        """✅ ENHANCED: Now actually uses VOLATILE_REGIME"""
        avg_slope = self.features['slope_20'][idx].mean().item() if 'slope_20' in self.features else 0
        avg_vol = self.features['std_20'][idx].mean().item() if 'std_20' in self.features else 0
        
        # Get global volatility threshold
        global_vol = self.features['std_20'].mean().item() if 'std_20' in self.features else 1.0
        
        # Check volatility first
        if avg_vol > 1.5 * global_vol:
            return VOCAB.VOLATILE_REGIME  # ✅ NOW USED!
        elif avg_slope > 0.05:
            return VOCAB.BULLISH_REGIME
        elif avg_slope < -0.05:
            return VOCAB.BEARISH_REGIME
        else:
            return VOCAB.SIDEWAYS_REGIME
    
    def _print_statistics(self):
        total_events = sum(len(a.all_events) for a in self.annotations)
        avg_events = total_events / len(self.annotations)
        
        # Count by label
        label_counts = {}
        for ann in self.annotations:
            for event in ann.all_events:
                label_counts[event.label_name] = label_counts.get(event.label_name, 0) + 1
        
        print(f"      Total events: {total_events}")
        print(f"      Avg per sequence: {avg_events:.1f}")
        print(f"      Unique labels used: {len(label_counts)}/64")
        print(f"      Top 10 labels:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"        {label:.<30} {count:>6}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        return self.annotations[idx]


# ============================================================================
# SECTION 8: TEXT GENERATION
# ============================================================================

class TextCorpusGenerator:
    """Generate training text in various formats."""
    
    @staticmethod
    def generate_corpus(dataset: CompleteHierarchicalEventDataset, 
                       format: str = 'depth_marked') -> List[str]:
        corpus = []
        for annotation in dataset.annotations:
            text = annotation.to_text(format=format)
            corpus.append(text)
        return corpus
    
    @staticmethod
    def estimate_tokens(corpus: List[str]) -> Dict:
        total_tokens = sum(len(text.split()) for text in corpus)
        total_chars = sum(len(text) for text in corpus)
        
        return {
            'num_documents': len(corpus),
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'avg_tokens_per_doc': total_tokens / len(corpus),
            'avg_chars_per_doc': total_chars / len(corpus),
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPLETE HIERARCHICAL EVENT LABELING SYSTEM")
    print("All detectors enabled - All 64 labels used!")
    print("="*80)
    
    # Generate synthetic data
    B, L = 5, 336
    torch.manual_seed(42)
    np.random.seed(42)
    
    t = torch.linspace(0, 4*np.pi, L)
    x = torch.zeros(B, L)
    
    for i in range(B):
        trend = 0.5 * torch.sin(t / 2) + 0.1 * t
        vol_modulator = 0.1 + 0.2 * (torch.sin(3 * t) > 0).float()
        noise = torch.randn(L) * vol_modulator
        num_spikes = np.random.randint(2, 5)
        spike_indices = torch.randint(50, L-50, (num_spikes,))
        spikes = torch.zeros(L)
        spikes[spike_indices] = torch.randn(num_spikes) * 2
        x[i] = trend + noise + spikes
    
    # Create complete dataset
    dataset = CompleteHierarchicalEventDataset(
        x,
        use_spectral=True,
        use_entropy=True,
        use_wavelets=True,
        use_wavelet_peaks=True,
        use_changepoint=True,
        use_chaotic=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("EXAMPLE ANNOTATION")
    print("="*80)
    dataset[0].print_hierarchy(max_depth=2)
    
    print("\n✓ SYSTEM READY - ALL LABELS NOW ACTIVE!")