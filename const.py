nchars = 28  # a,b,...,z plus ^, $ for start and end of word.
begin, end = '^', '$'

def get_itoa(nchars):
    out = [begin]
    for i in range(nchars-2):
        out.append(chr(ord('a')+i))
    out.append(end)
    return out

itoa = get_itoa(nchars)

def get_atoi(itoa):
    return {ch: i for i, ch in enumerate(itoa)}

atoi = get_atoi(itoa)
