import os
import numpy as np

widths = [
    (126,    1), (159,    0), (687,     1), (710,   0), (711,   1), 
    (727,    0), (733,    1), (879,     0), (1154,  1), (1161,  0), 
    (4347,   1), (4447,   2), (7467,    1), (7521,  0), (8369,  1), 
    (8426,   0), (9000,   1), (9002,    2), (11021, 1), (12350, 2), 
    (12351,  1), (12438,  2), (12442,   0), (19893, 2), (19967, 1),
    (55203,  2), (63743,  1), (64106,   2), (65039, 1), (65059, 0),
    (65131,  2), (65279,  1), (65376,   2), (65500, 1), (65510, 2),
    (120831, 1), (262141, 2), (1114109, 1),
]
 
def char_width( o ):
    global widths
    if o == 0xe or o == 0xf:
        return 0
    for num, wid in widths:
        if o <= num:
            return wid
    return 1

def sent_len(s):
    assert isinstance(s, str)
    ret = 0
    for it in s:
        ret += char_width(ord(it))
    return ret


def word_align(wordA, wordB):
    if sent_len(wordA) < sent_len(wordB):
        wordA += " " * (sent_len(wordB) - sent_len(wordA))
    else:
        wordB += " " * (sent_len(wordA) - sent_len(wordB))
    return wordA, wordB


def levenshtein_visual(a, b):
    la = len(a)
    lb = len(b)
    f = np.zeros((la + 1, lb + 1), dtype=np.uint64)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0:
                f[i][j] = j
            elif j == 0:
                f[i][j] = i
            elif a[i - 1].lower() == b[j - 1].lower():
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
    
    p, q = la, lb
    ret = []
    while p > 0 and q > 0:
        if a[p - 1].lower() == b[q - 1].lower():
            ret.append( (a[p - 1], b[q - 1]) )
            p -= 1
            q -= 1
        else:
            if f[p][q] == f[p - 1][q - 1] + 1:
                # modify
                ret.append( word_align(a[p - 1], b[q - 1]) )
                p -= 1
                q -= 1
            elif f[p][q] == f[p - 1][q] + 1:
                # remove
                ret.append( word_align(a[p - 1], "") )
                p -= 1
            else:
                assert f[p][q] == f[p][q - 1] + 1
                ret.append( word_align("", b[q - 1]) )
                q -= 1
    while p > 0:
        ret.append( word_align( a[p - 1], "" ) )
        p -= 1
    while q > 0:
        ret.append( word_align( "", b[q - 1] ) )
        q -= 1
    return ret[::-1]

def get_change(x_orig, x_adv):
    token_orig = x_orig.split()
    token_adv = x_adv.split()
    pairs = levenshtein_visual(token_orig, token_adv)
    ret = ""
    ret2 = ""
    curr1 = ""
    curr2 = ""
    for tokenA, tokenB in pairs:
        if tokenA.lower().strip() == tokenB.lower().strip():
            curr1 = tokenA + " "
            curr2 = tokenA + " "
        else:
            curr1 = "<b><font color='green'>" + tokenA + "</font></b>" + " "
            curr2 = "<b><font color='red'>" + tokenB + "</font></b>" + " "
        ret  += curr1
        ret2 += curr2
    return ret, ret2

