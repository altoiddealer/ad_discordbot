from math import sqrt, gcd

from modules.utils_files import load_file
from modules.utils_shared import shared_path

from modules.logs import import_track, get_logger; import_track(__file__, fp=True); log = get_logger(__name__)  # noqa: E702
logging = log

def round_to_precision(val, prec):
    return round(val / prec) * prec

def res_to_model_fit(avg, w, h, prec):
    mp = w * h
    mp_target = avg * avg
    scale = sqrt(mp_target / mp)
    w = int(round_to_precision(w * scale, prec))
    h = int(round_to_precision(h * scale, prec))
    return w, h

def dims_from_ar(avg, n, d):
    from modules.utils_shared import config
    ROUNDING_PRECISION = config.imggen.get('rounding_precision', 64)
    doubleavg = avg * 2
    ar_sum = n+d
    # calculate width and height by factoring average with aspect ratio
    w = round((n / ar_sum) * doubleavg)
    h = round((d / ar_sum) * doubleavg)
    # Round to correct megapixel precision
    w, h = res_to_model_fit(avg, w, h, ROUNDING_PRECISION)
    return w, h

def avg_from_dims(w, h):
    avg = (w + h) // 2
    if (w + h) % 2 != 0:
        avg += 1
    return avg

def init_avg_from_dims():
    base_settings = load_file(shared_path.base_settings, {})
    w = base_settings.get('imgmodel', {}).get('payload', {}).get('width', 512)
    h = base_settings.get('imgmodel', {}).get('payload', {}).get('height', 512)
    return avg_from_dims(w, h)

def ar_parts_from_dims(w, h):
    divisor = gcd(w, h)
    simp_w = w // divisor
    simp_h = h // divisor
    return simp_w, simp_h

def get_aspect_ratio_parts(ratio):
    try:
        ratio_parts = tuple(map(int, ratio.replace(':', '/').split('/')))
        return ratio_parts[0], ratio_parts[1]
    except Exception as e:
        log.error(f'Could not split ratio "{ratio}" into parts: {e}')
        return None, None

def calculate_aspect_ratio_sizes(avg, aspect_ratios):
    ratio_options = []
    try:
        for ratio in aspect_ratios:
            n, d = get_aspect_ratio_parts(ratio)
            w, h = dims_from_ar(avg, n, d)
            # Apply labels
            if w > h: 
                aspect_type = "landscape"
            elif w < h: 
                aspect_type = "portrait"
            else: 
                aspect_type = "square"
            # Format the result
            size_name = f"{w} x {h} ({ratio} {aspect_type})"
            ratio_options.append({'name': size_name, 'width': w, 'height': h})
    except Exception as e:
        log.error(f'Error while calculating aspect ratio sizes for "/image" cmd: {e}')
    return ratio_options
