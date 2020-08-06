from utils import format_multi_line
import numpy as np

MAXBITS = 15
MAXLCODES = 286
MAXDCODES = 30
MAXDCODES = MAXDCODES + MAXLCODES
FIXLCODES = 288
MAXDIST = 32768


class State:
    def __init__(self, stream):
        self.bitbuf = 0
        self.bitcnt = 0

        self.stream = stream
        self.stream_idx = 0
        self.uncompressed = b''

    def getc(self):
        self.stream_idx += 1
        return self.stream[self.stream_idx - 1]


class Huffman:
    """
    Huffman decoding tables.
    count[1 ... MAXBITS] is the number of symbols of each length
    symbol[] are the symbol values in canonical order. The total number of entries is the sum of counts
    """

    def __init__(self, num_codes):
        self.count = np.zeros(shape=MAXBITS + 1, dtype=int)  # number of symbols of each length
        self.symbol = np.zeros(shape=num_codes, dtype=int)  # canonically ordered symbols

    def construct(self, length, n):
        """
        Construct Huffman code for n symbols. Tables are the number of codes of each length, and the symbols sorted by
        length, retaining their original order within each length.
        :param length:  The length of each code representing the symbol
        :param n: The number of code words
        :return: None
        """
        # count the number of codes of each length
        for l in length:
            self.count[l] += 1
        if self.count[0] == n:
            raise ValueError('Incomplete Huffman tree')

        # generate offsets into symbol table for each length for sorting
        offs = np.zeros(MAXBITS + 1, dtype=int)
        for l in range(MAXBITS):
            offs[l + 1] = offs[l] + self.count[l]

        # put symbols in table sorting by length, symbol order within each length
        for symbol in range(n):
            if length[symbol] != 0:
                self.symbol[offs[length[symbol]]] = symbol  # insert symbol
                offs[length[symbol]] += 1  # sort based on length

    def __str__(self):
        string = ""
        for s in self.symbol:
            string += "{} \n".format(s)

        return string


class INFLATE:
    """
    This class is to decode zlib/DEFLATE stream into readable literals and (length, distance) tokens
    """

    def __init__(self):
        self.s = None
        self.uncompressed_msg = b''
        self.alder32 = None
        self.infgen = ""
        self.remaining_bytes = None

    def decompress(self, stream):
        self.s = State(stream)
        self.uncompressed_msg = b''
        self.alder32 = None

        trail = 0

        # check and process headers if any
        ret = self.s.getc()
        n = self.s.getc()
        val = (ret << 8) + n

        if val % 31 == 0 and (ret & 15) == 8 and (ret >> 4) < 8:
            # zlib headers
            trail = 4
            self.infgen_log("zlib")

        else:
            # raw DEFLATE
            self.s.stream_idx = 0  # reset stream indexing
            self.infgen_log("raw DEFLATE")

        # inflate deflate stream
        self.inflate()

        if trail == 4:
            self.alder32 = self.s.stream[self.s.stream_idx:self.s.stream_idx + 4]
            self.infgen_log("alder")

        self.remaining_bytes = self.s.stream[self.s.stream_idx + trail:]

    def inflate(self):
        last = 0
        while not last:
            # Verify block
            last = self.bits(1)

            self.infgen_log("!")
            if last and self.s.stream_idx != len(self.s.stream):
                self.infgen_log("last block")

            # Verify mode
            mode = self.bits(2)

            if mode == 0:
                self.infgen_log('mode 0 : stored')
                self.stored()
            elif mode == 1:
                self.infgen_log('mode 1 : fixed')
                self.fixed()

            elif mode == 2:
                self.infgen_log('mode 2 : dynamic')
                raise ValueError('DEFLATE mode 2 has not been implemented')
            else:
                raise ValueError('DEFLATE mode undefined')
        return 1

    def bits(self, need):
        """
        Returns the number of bits needed from input stream.
        :param state: state of stream and buffer
        :param need: number of bits needed
        :return: needed bits
        """
        val = self.s.bitbuf
        while self.s.bitcnt < need:
            next_byte = self.s.stream[self.s.stream_idx]
            self.s.stream_idx += 1
            if self.s.stream_idx >= len(self.s.stream):
                raise ValueError('Out of inputs')
            else:
                self.s.bitbuf = next_byte
                val = val | next_byte << self.s.bitcnt
                self.s.bitcnt += 8

        # drop number of needed bits and update buffer
        self.s.bitbuf = val >> need
        self.s.bitcnt -= need

        # return needed bits, zeroing the other bits above that
        return val & (2 ** need - 1)

    def fixed(self):
        self.lencode = Huffman(FIXLCODES)
        self.distcode = Huffman(MAXDCODES)

        # Construct Huffman Table for literals and lengths
        lengths = np.zeros(FIXLCODES, dtype=int)
        lengths[0: 144] = 8
        lengths[144: 256] = 9
        lengths[256: 280] = 7
        lengths[280: FIXLCODES] = 8

        self.lencode.construct(lengths, FIXLCODES)

        # Construct Huffman Table for distance
        lengths = np.ones(MAXDCODES, dtype=int) * 5
        self.distcode.construct(lengths, MAXDCODES)

        return self.codes()

    def stored(self):
        """
        flush the current block for to byte boundary
        :return:
        """
        # discard leftover bits from current byte
        self.s.bitcnt = 0
        self.s.bitbuf = 0

        # get length and check against complement
        length = self.s.getc()
        length += self.s.getc() << 8
        cmp = self.s.getc()
        octet = self.s.getc()
        cmp += octet << 8

        if cmp != 0xffff:
            raise ValueError("Complement not matched")

        self.infgen_log("end")

    def decode(self, huffman):
        """
        Read in one bit at a time and traverse though Huffman tree to find corresponding symbol
        :param huffman: Huffman Tree to perform decoding on
        :return: decoded symbol from Huffman Tree
        """
        # Run though remaining bit stream to uncompress data
        bitbuf = self.s.bitbuf
        left = self.s.bitcnt
        code, first, index = 0, 0, 0
        length = 1
        next = 1

        while True:
            while left:
                left -= 1
                code = code | (bitbuf & 1)  # read in 1 bit at a time
                bitbuf = bitbuf >> 1  # update buffer
                count = huffman.count[next]
                next += 1
                if code < first + count:
                    # code is valid within the Huffman tree, so we return the corresponding symbol here
                    self.s.bitbuf = bitbuf
                    self.s.bitcnt = left
                    return huffman.symbol[index + (code - first)]
                else:
                    # update for the next bit
                    index += count
                    first += count
                    first = first << 1
                    code = code << 1
                    length += 1

            # read in one more btye
            bitbuf = self.s.stream[self.s.stream_idx]
            self.s.stream_idx += 1
            left = 8

    def codes(self):
        """
        Decode literal/length and distance codes until the end of block
        :return: None
        """
        lens = np.array([  # Size base for length codes 257..285
            3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
            35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258])
        lext = np.array([  # Extra bits for length codes 257..285
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
            3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0])
        dists = np.array([  # Offset base for distance codes 0..29
            1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
            257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
            8193, 12289, 16385, 24577])
        dext = np.array([  # Extra bits for distance codes 0..29
            0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
            12, 12, 13, 13])

        symbol = 0  # decoded symbol

        # decode literals anf length/distance pairs
        while symbol != 256:
            symbol = self.decode(self.lencode)
            if symbol < 0:
                raise ValueError("symbol must be non-negative")
            elif symbol < 256:
                # write literals to uncompress bit streams
                self.uncompressed_msg += bytes([symbol])
                self.infgen_log('literal : {}'.format(hex(symbol)))

            elif symbol > 256:
                # get and compute length
                if symbol >= MAXLCODES:
                    raise ValueError("Invalid fixed code")

                symbol -= 257
                length = lens[symbol] + self.bits(lext[symbol])

                # get distance
                symbol = self.decode(self.distcode)
                if symbol < 0:
                    raise ValueError("Invalid distance symbol")

                dist = dists[symbol] + self.bits(dext[symbol])

                self.copy(length=length, distance=dist)
                self.infgen_log("match {} {}".format(length, dist))

            elif symbol == 256:
                self.infgen_log("end")

    def copy(self, distance, length):
        d = len(self.uncompressed_msg) - distance                   # number of places to go back
        cycles = length // distance
        rem_elements = length % distance
        copy_full_length = self.uncompressed_msg[d:] * cycles       # circular copy number of rounds
        self.uncompressed_msg += copy_full_length + self.uncompressed_msg[d: d + rem_elements]

    def infgen_log(self, string):
        self.infgen += string
        self.infgen += '\n'


if __name__ == '__main__':
    original_data = b'\x11' * 9 + b'\x22' * 10 + b'\x33' * 291 + \
                    b'\x11\x12\x11\x13\x14\x11\x14\x14\x11\x14\x14\x11\x12' * 3 + b'\x00' * 5
    compress_data = b'\x13\x14\x84\x02\x25\x38\x30\x1E\x05\x84\x42\x40\x50\x48\x50\x58' \
                    b'\x44\x50\x04\x8C\x84\x70\x73\x18\x40\x00\x00'

    inflate = INFLATE()
    inflate.decompress(compress_data)
    uncompressed_msg = inflate.uncompressed_msg

    print('Compressed data :')
    print(format_multi_line('\t', compress_data, size=64))
    print()
    print('Infgen log:')
    print(inflate.infgen)
    if uncompressed_msg == original_data:
        print('=' * 64)
        print('Inflate successful')
        print('=' * 64)
    print('Uncompressed data :')
    print(format_multi_line('\t', uncompressed_msg, size=64))
    if inflate.remaining_bytes:
        print('Remaining btyes :')
        print(format_multi_line('\t', inflate.remaining_bytes, size=64))
