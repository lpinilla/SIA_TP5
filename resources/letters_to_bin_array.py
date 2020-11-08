import pickle

letters = []
letters.append("00000000000000") #20: space
letters.append("04040404040004") #21: !
letters.append("09091200000000") #22: "
letters.append("0a0a1f0a1f0a0a") #23: #
letters.append("040f140e051e04") #24: $
letters.append("19190204081313") #25: %
letters.append("040a0a0a15120d") #26: &
letters.append("04040800000000") #27: '
letters.append("02040808080402") #28: (
letters.append("08040202020408") #29: )
letters.append("04150e1f0e1504") #2a: *
letters.append("0004041f040400") #2b: +
letters.append("00000000040408") #2c: ,
letters.append("0000001f000000") #2d: -
letters.append("00000000000c0c") #2e: .
letters.append("01010204081010") #2f: /
letters.append("0e11131519110e") #30: 0
letters.append("040c040404040e") #31: 1
letters.append("0e11010204081f") #32: 2
letters.append("0e11010601110e") #33: 3
letters.append("02060a121f0202") #34: 4
letters.append("1f101e0101110e") #35: 5
letters.append("0608101e11110e") #36: 6
letters.append("1f010204080808") #37: 7
letters.append("0e11110e11110e") #38: 8
letters.append("0e11110f01020c") #39: 9
letters.append("000c0c000c0c00") #3a: :
letters.append("000c0c000c0408") #3b: ;
letters.append("02040810080402") #3c: <
letters.append("00001f001f0000") #3d: =
letters.append("08040201020408") #3e: >
letters.append("0e110102040004") #3f: ?

bin_letters = []

num_of_bits = 8 * 7
for i in range(len(letters)):
    bin_letters.append(bin(int(letters[i], 16))[2:].zfill(num_of_bits))

#remover los primeros 3 bits de cada byte
bin_letters_trimmed = []

for r in range(len(bin_letters)):
    exp = []
    for i in range(len(bin_letters[r])):
        if (i % 8) == 0:
            exp.append(bin_letters[r][(i + 3): (i + 8)])
    bin_letters_trimmed.append("".join(exp))

#pasar los strings a arrays de caracteres (como int)
letters_as_array_of_bits = []
for i in range(len(bin_letters_trimmed)):
    letters_as_array_of_bits.append([int(j) for j in list(bin_letters_trimmed[i])])

#guardar las letras en un pickle
f = open('letras.pickle', 'wb')
pickle.dump(letters_as_array_of_bits, f)
