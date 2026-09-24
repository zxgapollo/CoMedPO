import math

# Unified TIE metrics and DPO weight computation entry

def normalize_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().strip().split())

def jaccard_sim(a, b):
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0

def _get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def extract_tie_metrics(source):
    """
    Robustly extract TIE-related metrics from a dict-like source.
    Returns a dict with keys: delta_pos, delta_neg, m_v, m_n, gamma
    Missing values will be set to None.
    """
    return {
        'delta_pos': _get(source, ['delta_pos','delta_plus','Delta_plus','DeltaPos']),
        'delta_neg': _get(source, ['delta_neg','delta_minus','Delta_minus','DeltaNeg']),
        'm_v': _get(source, ['m_v','mv','visual_score']),
        'm_n': _get(source, ['m_n','mn','noise_score']),
        'gamma': _get(source, ['gamma','Gamma'])
    }

def is_valid(metrics):
    """
    Basic validity check for a sample given metrics.
    - gamma >= 0 (if available)
    - m_v >= 0.2 (if available)
    Missing values do not invalidate the sample.
    """
    gamma = metrics.get('gamma')
    m_v = metrics.get('m_v')
    if gamma is not None and gamma < 0:
        return False
    if m_v is not None and m_v < 0.2:
        return False
    return True

def compute_weight(
    gt_answer,
    pref_answer,
    disp_answer,
    metrics=None,
    w_min=0.05,
    beta=2.0,
    tau=0.0,
    # Scheme-2 (merged) weighting params
    w_gamma=1.0,
    w_v=0.5,
    w_n=0.8,
    w_s=0.3,
    w_o=0.5,
    epsilon=0.02,
    tie_diff=None,
):
    """
    Unified weight computation entry aligned to Scheme-2 (merged) logic.

    If TIE metrics are provided, compute weight using Scheme-2 formula:
        weighted_score = w_gamma*gamma + w_v*m_v + w_n*m_n + w_s*tie_diff
                         + w_o*(delta_pos - delta_neg)
        sigmoid_input   = beta * (weighted_score - tau)
        weight          = sigmoid(sigmoid_input - epsilon)
        final           = max(w_min, weight)

    - gamma: prefer m_v - m_n when not explicitly provided
    - tie_diff: optional (defaults to 0 if unavailable)

    Otherwise, fallback to text similarity margin against GT (Jaccard),
    preserving existing behavior and parameters (w_min, beta, tau).
    """

    def _sigmoid(x):
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    # Metric-driven weighting (preferred)
    dp = metrics.get('delta_pos') if metrics else None
    dn = metrics.get('delta_neg') if metrics else None
    mv = metrics.get('m_v') if metrics else None
    mn = metrics.get('m_n') if metrics else None
    gamma = metrics.get('gamma') if metrics else None

    # Try to resolve tie_diff from metrics if not explicitly provided
    if tie_diff is None and isinstance(metrics, dict):
        tpos = metrics.get('tie_positive') or metrics.get('tie_pos') or metrics.get('tie_pos_token_avg')
        tneg = metrics.get('tie_negative') or metrics.get('tie_neg') or metrics.get('tie_neg_token_avg')
        if tpos is not None and tneg is not None:
            try:
                tie_diff = float(tpos) - float(tneg)
            except Exception:
                tie_diff = None

    if dp is not None or dn is not None or mv is not None or mn is not None or gamma is not None:
        dp = float(dp) if dp is not None else 0.0
        dn = float(dn) if dn is not None else 0.0
        mv = float(mv) if mv is not None else 0.0
        mn = float(mn) if mn is not None else 0.0
        if gamma is None:
            gamma = float(mv - mn)
        else:
            gamma = float(gamma)
        td = float(tie_diff) if tie_diff is not None else 0.0

        weighted_score = (
            w_gamma * gamma
            + w_v * mv
            + w_n * mn
            + w_s * td
            + w_o * (dp - dn)
        )
        sigmoid_input = beta * (weighted_score - tau)
        base = _sigmoid(sigmoid_input - epsilon)
        return round(max(w_min, float(base)), 6)

    # Fallback: similarity margin w.r.t GT
    sim_pref = jaccard_sim(gt_answer or '', pref_answer or '')
    sim_disp = jaccard_sim(gt_answer or '', disp_answer or '')
    margin = max(0.0, sim_pref - sim_disp - tau)
    base = min(1.0, beta * margin)
    return round(w_min + (1.0 - w_min) * base, 6)