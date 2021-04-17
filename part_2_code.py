from math import log2
import pandas as pd
import numpy as np


r = lambda n: 4/5 * n
S = lambda n: 5/4 * n**2
N = lambda n: S(n) * (n - r(n))
L = lambda n: n**3


def h_bar_fat(n): 
    return ((n/2)**4 * (n-1) * 6 + (n/2)**3 * (n/2-1) * 4 + (n/2)**2 * (n/2-1) * 2) / ((n/2)**4 * (n-1) + (n/2)**3 * (n/2-1) + (n/2)**2 * (n/2-1))


def h_bar_jelly_bound(n):
    k = lambda n: 1 + int( ( log2( S(n) - ( 2 * (S(n) - 1) / r(n) ) ) ) / log2( r(n) - 1 ) )
    R = lambda n: S(n) - 1 - sum([r(n) * (r(n) - 1)**(j-1) for j in range(1, k(n))])
    return (sum([j * r(n) * (r(n) - 1)**(j-1) for j in range(1, k(n))]) + k(n) * R(n)) / (S(n) - 1)


def throughput_fat(n):
    l = lambda n:  n**3
    nu_f = lambda n: n * ((n/2)**4*(n-1) + (n/2)**3*(n/2-1) + (n/2)**2*(n/2-1))
    return l(n) / (h_bar_fat(n) * nu_f(n))
    

def throughput_jelly(n):
    l = lambda n: S(n) * r(n)
    nu_f = lambda n: N(n) * (N(n) - 1)
    return l(n) / (h_bar_jelly_bound(n) * nu_f(n))


def format_tex(float_number):
    exponent = np.floor(np.log10(float_number))
    mantissa = float_number/10**exponent
    mantissa_format = str(mantissa)[0:3]
    return "${0}\times10^{{{1}}}$"\
           .format(mantissa_format, str(int(exponent)))



# print()
# n = int(input("Number of ports of a switch:\t"))

# print(f'Average path length in a fat tree having n={n}:\t{h_bar_fat(n)}')

# print()    

# print(f'Throughput Fat-Tree:\t{throughput_fat(n)}\n\n')

# print(f'Throughput Jellyfish:\t{throughput_jelly(n)}\n\n')

# print(f'N:{N(n)}, S:{S(n)}\n')
 

results = {'n':[], 'N':[], 'S':[], 'L':[], 't_f':[], 't_j':[]}
for n in range(10, 51, 10):
    results['n'].append(n)
    results['N'].append(int(N(n)))
    results['S'].append(int(S(n)))
    results['L'].append(int(L(n)))
    results['t_f'].append(throughput_fat(n))
    results['t_j'].append(throughput_jelly(n))

df = pd.DataFrame(results)
df['t_f'] = df['t_f'].map(lambda x:format_tex(x))
df['t_j'] = df['t_j'].map(lambda x:format_tex(x))
print(df.to_latex(index=False, escape=False))
