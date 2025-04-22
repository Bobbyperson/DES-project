import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

# fmt: off
# ---------------------------------------------------------------------------#
# ────────────────  DES core (initial/ final perm, E, S, P)  ────────────────#
IP  = [58,50,42,34,26,18,10,2, 60,52,44,36,28,20,12,4,
       62,54,46,38,30,22,14,6, 64,56,48,40,32,24,16,8,
       57,49,41,33,25,17,9,1,  59,51,43,35,27,19,11,3,
       61,53,45,37,29,21,13,5, 63,55,47,39,31,23,15,7]

FP  = [40,8,48,16,56,24,64,32, 39,7,47,15,55,23,63,31,
       38,6,46,14,54,22,62,30, 37,5,45,13,53,21,61,29,
       36,4,44,12,52,20,60,28, 35,3,43,11,51,19,59,27,
       34,2,42,10,50,18,58,26, 33,1,41,9,49,17,57,25]

EXP = [32,1,2,3,4,5, 4,5,6,7,8,9, 8,9,10,11,12,13, 12,13,14,15,16,17,
       16,17,18,19,20,21, 20,21,22,23,24,25, 24,25,26,27,28,29, 28,29,30,31,32,1]

PBOX= [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,
       2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]

# 8 S-boxes from FIPS-46
SBOX = [
    # S1
    [[14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
     [0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],
     [4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
     [15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13]],
    # S2
    [[15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],
     [3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5],
     [0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15],
     [13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9]],
    # S3
    [[10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],
     [13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
     [13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],
     [1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12]],
    # S4
    [[7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],
     [13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],
     [10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],
     [3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14]],
    # S5
    [[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],
     [14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],
     [4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],
     [11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3]],
    # S6
    [[12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],
     [10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],
     [9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],
     [4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13]],
    # S7
    [[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],
     [13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],
     [1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],
     [6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12]],
    # S8
    [[13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],
     [1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],
     [7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],
     [2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11]]
]
# ─────────────────────────────────────────────────────────────────────────── #

# fmt: on
def permute(bits: int, table: list[int], width: int) -> int:
    out = 0
    for pos in table:
        out = (out << 1) | ((bits >> (width - pos)) & 1)
    return out


def expand(r32: int) -> int:
    return permute(r32, EXP, 32)


def sbox_sub(i: int, six_bits: int) -> int:
    row = ((six_bits & 0b100000) >> 4) | (six_bits & 1)
    col = (six_bits >> 1) & 0b1111
    return SBOX[i][row][col]


def f_func(r32: int, k48: int) -> int:
    e = expand(r32) ^ k48
    out = 0
    for i in range(8):
        s_in = (e >> (42 - 6 * i)) & 0x3F  # leftmost block first
        out = (out << 4) | sbox_sub(i, s_in)
    return permute(out, PBOX, 32)


def des4_encrypt_block(p64: int, subkeys):
    ip = permute(p64, IP, 64)
    L, R = ip >> 32, ip & 0xFFFFFFFF

    # rounds 1-3
    for k in subkeys[:3]:
        L, R = R, L ^ f_func(R, k)

    L3, R3 = L, R  # save for attack
    # round 4
    L4, R4 = R3, L3 ^ f_func(R3, subkeys[3])
    pre_output = (L4 << 32) | R4
    ciph = permute(pre_output, FP, 64)
    return ciph, L3, R3


# fmt: off
PC1 = [57,49,41,33,25,17,9,1,58,50,42,34,26,18,
       10,2,59,51,43,35,27,19,11,3,60,52,44,36,
       63,55,47,39,31,23,15,7,62,54,46,38,30,22,
       14,6,61,53,45,37,29,21,13,5,28,20,12,4]

PC2 = [14,17,11,24,1,5,3,28,15,6,21,10,23,19,
       12,4,26,8,16,7,27,20,13,2,41,52,31,37,
       47,55,30,40,51,45,33,48,44,49,39,56,34,53,
       46,42,50,36,29,32]

ROT = [1,1,2,2]  # first four rotations
# fmt: on


def gen_round_keys(key64_hex: str):
    key64 = int(key64_hex, 16)
    key56 = permute(key64, PC1, 64)
    c = (key56 >> 28) & 0xFFFFFFF
    d = key56 & 0xFFFFFFF
    subkeys = []
    for i in range(4):  # only 4 rounds
        rot = ROT[i]
        mask = (1 << 28) - 1
        c = ((c << rot) | (c >> (28 - rot))) & mask
        d = ((d << rot) | (d >> (28 - rot))) & mask
        cd = (c << 28) | d
        subkeys.append(permute(cd, PC2, 56))
    return subkeys


DELTA_P = 0x0800000000000000
PAIR_NUM = 100_000


def attack_round4_key(key_hex: str, pairs=PAIR_NUM):
    print(f"[*] Encrypting {pairs:,} chosen pairs …")
    subkeys = gen_round_keys(key_hex)
    results = []

    for _ in tqdm(range(pairs), unit="pair"):
        P = random.getrandbits(64)
        C1, L3a, R3a = des4_encrypt_block(P, subkeys)
        C2, L3b, R3b = des4_encrypt_block(P ^ DELTA_P, subkeys)
        results.append((C1, C2, L3a, R3a, L3b, R3b))

    print("[*] Scoring 6-bit candidates for each S-box …")
    best_subkey = []
    for sbox in range(8):
        counter = Counter()
        for C1, C2, L3a, R3a, L3b, R3b in tqdm(results, unit="pair", leave=False):
            e1 = expand(R3a)
            e2 = expand(R3b)
            blk1 = (e1 >> (42 - 6 * sbox)) & 0x3F
            blk2 = (e2 >> (42 - 6 * sbox)) & 0x3F
            for k in range(64):
                if sbox == 0:  # characteristic hits only S1
                    if sbox_sub(sbox, blk1 ^ k) ^ sbox_sub(sbox, blk2 ^ k) == 0:
                        counter[k] += 1
                else:  # inactive boxes
                    counter[k] += 1
        best_subkey.append(max(counter, key=counter.get))
    return best_subkey


if __name__ == "__main__":
    true_key = random.getrandbits(56) << 8  # ignore parity
    key_hex = f"{true_key:016X}"
    print(f"\nTrue 64-bit key (hex, parity-free): {key_hex}")

    recovered = attack_round4_key(key_hex)
    print("\nRecovered sub-key (round 4): " + " ".join(f"{k:02X}" for k in recovered))

    truth_chunks = [
        (gen_round_keys(key_hex)[3] >> (42 - 6 * i)) & 0x3F for i in range(8)
    ]
    correct = sum(r == t for r, t in zip(recovered, truth_chunks))
    print(f"[+] Correct 6-bit chunks: {correct}/8")
