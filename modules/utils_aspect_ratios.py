from ad_discordbot.modules.logs import import_track, log, get_logger; import_track(__file__, fp=True)
from math import sqrt
log = get_logger(__name__)
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

def dims_from_ar(avg, n, d, prec=64):
    doubleavg = avg * 2
    ar_sum = n+d
    # calculate width and height by factoring average with aspect ratio
    w = round((n / ar_sum) * doubleavg)
    h = round((d / ar_sum) * doubleavg)
    # Round to correct megapixel precision
    w, h = res_to_model_fit(avg, w, h, prec)
    return w, h

def avg_from_dims(w, h):
    avg = (w + h) // 2
    if (w + h) % 2 != 0:
        avg += 1
    return avg

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
            if w > h: aspect_type = "landscape"
            elif w < h: aspect_type = "portrait"
            else: aspect_type = "square"
            # Format the result
            size_name = f"{w} x {h} ({ratio} {aspect_type})"
            ratio_options.append({'name': size_name, 'width': w, 'height': h})
    except Exception as e:
        log.error(f'Error while calculating aspect ratio sizes for "/image" cmd: {e}')
    return ratio_options
