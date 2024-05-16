import struct
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid32(x):
    return float(np.float32(1. / (1. + expf(np.float32(-x)))))


def softmax(x):
    x = np.exp(x)
    return (x.T / x.sum(axis=(1 if x.ndim == 2 else 0))).T


def softmax32(x):
    x_max = max(x)
    norm = 0.
    for i, t in enumerate(x):
        x[i] = expf(t - x_max)
        norm += np.float32(x[i])
    norm = np.float32(norm)
    return [float(np.float32(t) / norm) for t in x]


def relu(x):
    return np.maximum(x, 0)


def identity(x):
    return x


#########################################################################################
# Code below to reproduce the expf function used in XGBoost as done in DSS Java scoring #
#########################################################################################

EXP2F_TABLE_BITS = 5
N = 32  # 2^EXP2F_TABLE_BITS

# Constants defined in glibc e_exp2f_data.c
InvLn2N = (float.fromhex('0x1.71547652b82fep+0')) * N  # N/log(2)

# Encode 2^(k/N) for k in [0, N-1], as a long value
# whose bits are those of the double approximation of 2^(k/N) - (minus) those of the long 2^52*k/N
T = [
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540
]
# Coefficients of the 3rd degree Taylor approximation of 2^x close to 0
C = [
    float.fromhex('0x1.c6af84b912394p-5') / N / N / N,
    float.fromhex('0x1.ebfce50fac4f3p-3') / N / N,
    float.fromhex('0x1.62e42ff0c52d6p-1') / N
]


def expf(x):
    # x*N/Ln2 = k + r with r in [-1/2, 1/2] and k integer
    z = InvLn2N * x
    kd = round(z)
    ki = int(kd)
    r = z - kd
    # Use T to approximate 2^(k/N)
    index = ki % N if ki % N >= 0 else ki % N + N
    t = T[index]  # Set the fractional part of s as encoded in T
    t += ki << (52 - EXP2F_TABLE_BITS)  # Set the exponent part of s as [ki/N]
    # Read t's 64 bits (long) as a (double) representation approximation of s = 2^(k/N)
    s = struct.unpack('d', struct.pack('Q', t))[0]
    # Compute 2^(r/N) ~= C0*r^3 + C1*r^2 + C2*r + 1
    z = C[0] * r + C[1]
    r2 = r * r
    y = C[2] * r + 1
    y = z * r2 + y
    # exp(x) = 2^(r/N) * 2^(k/N) ~= (C0*r^3 + C1*r^2 + C2*r + 1) * s
    y = y * s
    return np.float32(y)
