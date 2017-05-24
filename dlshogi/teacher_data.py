import shogi
from ctypes import *

# see: https://github.com/mk-takizawa/elmo_for_learn
#      position.cpp:33
boardCodeTable = {
    ''  : (0b0, 1), # Empty
    'P' : (0b1, 4), # BPawn
    'L' : (0b11, 6), # BLance
    'N' : (0b111, 6), # BKnight
    'S' : (0b1011, 6), # BSilver
    'B' : (0b11111, 8), # BBishop
    'R' : (0b111111, 8), # BRook
    'G' : (0b1111, 6), # BGold
    'K' : (0b0, 0), # BKing 玉の位置は別途、位置を符号化する。使用しないので numOfBit を 0 にしておく。
    '+P': (0b1001, 4), # BProPawn
    '+L': (0b100011, 6), # BProLance
    '+N': (0b100111, 6), # BProKnight
    '+S': (0b101011, 6), # BProSilver
    '+B': (0b10011111, 8), # BHorse
    '+R': (0b10111111, 8), # BDragona
    # (0b0, 0), 使用しないので numOfBit を 0 にしておく。
    # (0b0, 0), 使用しないので numOfBit を 0 にしておく。
    'p' : (0b101, 4), # WPawn
    'l' : (0b10011, 6), # WLance
    'n' : (0b10111, 6), # WKnight
    's' : (0b11011, 6), # WSilver
    'b' : (0b1011111, 8), # WBishop
    'r' : (0b1111111, 8), # WRook
    'g' : (0b101111, 6), # WGold
    'k' : (0b0, 0), # WKing 玉の位置は別途、位置を符号化する。
    '+p': (0b1101, 4), # WProPawn
    '+l': (0b110011, 6), # WProLance
    '+n': (0b110111, 6), # WProKnight
    '+s': (0b111011, 6), # WProSilver
    '+b': (0b11011111, 8), # WHorse
    '+r': (0b11111111, 8), # WDragon
}

# 盤上の bit 数 - 1 で表現出来るようにする。持ち駒があると、盤上には Empty の 1 bit が増えるので、
# これで局面の bit 数が固定化される。
handCodeTable = {
    shogi.PAWN   : ((0b0, 3), (0b100, 3)), # HPawn
    shogi.LANCE  : ((0b1, 5), (0b10001, 5)), # HLance
    shogi.KNIGHT : ((0b11, 5), (0b10011, 5)), # HKnight
    shogi.SILVER : ((0b101, 5), (0b10101, 5)), # HSilver
    shogi.GOLD   : ((0b111, 5), (0b10111, 5)), # HGold
    shogi.BISHOP : ((0b11111, 7), (0b1011111, 7)), # HBishop
    shogi.ROOK   : ((0b111111, 7), (0b1111111, 7)), # HRook
}

# see: piece.hpp:32: enum PieceType
PieceType = [
    shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.BISHOP, shogi.ROOK, shogi.GOLD, shogi.KING
]

# see: position.hpp:161: static void init()
boardCodeToPieceHash = {}
handCodeToPieceHash = {}
def toKey(code, numOfBits):
    return (numOfBits << 8) + code

for key, val in boardCodeTable.items():
    pc = shogi.Piece.from_symbol(key)
    if pc.piece_type != shogi.KING: # 玉は位置で符号化するので、駒の種類では符号化しない。
        boardCodeToPieceHash[toKey(val[0], val[1])] = pc
for key, val in handCodeTable.items():
    for c in shogi.COLORS:
        pc = shogi.Piece(key, c)
        handCodeToPieceHash[toKey(val[c][0], val[c][1])] = pc

# see: position.hpp:178: struct HuffmanCodedPosAndEval
class HuffmanCodedPosAndEval(Structure):
    _fields_ = (
        ('hcp', c_byte * 32),
        ('eval', c_short),
        ('bestMove16', c_ushort),
        ('gameResult', c_byte),
    )

class BitStream(object):
    # 読み込むデータをセットする。
    def __init__(self, d):
        self.data = d
        self.p = 0
        self.curr = 0
    # １ bit 読み込む。どこまで読み込んだかを表す bit の位置を 1 個進める。
    def getBit(self):
        result = 1 if self.data[self.p] & (1 << self.curr) > 0 else 0
        self.curr += 1
        if self.curr == 8:
            self.p += 1
            self.curr = 0
        return result
    # numOfBits bit読み込む。どこまで読み込んだかを表す bit の位置を numOfBits 個進める。
    def getBits(self, numOfBits):
        result = 0
        for i in range(numOfBits):
            result |= self.getBit() << i
        return result

# see: position.cpp:2008: bool Position::set(const HuffmanCodedPos& hcp, Thread* th)
#      position.hpp:156:  struct HuffmanCodedPos
def decode_hcpe(hcpe):
    # HuffmanCodedPos hcp
    bs = BitStream(hcpe.hcp)

    board = shogi.Board()
    board.clear()

    # 手番
    board.turn = bs.getBit()

    # 玉の位置
    sq0 = bs.getBits(7)
    sq1 = bs.getBits(7)
    board.set_piece_at(shogi.SQUARES_L90[sq0], shogi.Piece(shogi.KING, shogi.BLACK))
    board.set_piece_at(shogi.SQUARES_L90[sq1], shogi.Piece(shogi.KING, shogi.WHITE))

    # 盤上の駒
    for sq in shogi.SQUARES_L90:
        if board.piece_at(sq) is not None and board.piece_at(sq).piece_type == shogi.KING: # piece(sq) は BKing, WKing, Empty のどれか。
            continue
        numOfBits = 0
        code = 0
        while numOfBits <= 8:
            code |= bs.getBit() << numOfBits
            numOfBits += 1
            key = toKey(code, numOfBits)
            if key in boardCodeToPieceHash:
                pc = boardCodeToPieceHash[key]
                board.set_piece_at(sq, pc)
                break
    while bs.p < len(bs.data):
        numOfBits = 0
        code = 0
        while numOfBits <= 8:
            code |= bs.getBit() << numOfBits
            numOfBits += 1
            key = toKey(code, numOfBits)
            if key in handCodeToPieceHash:
                pc = handCodeToPieceHash[key]
                board.add_piece_into_hand(pc.piece_type, pc.color)
                break

    # u16 bestMove16
    bestMove16 = hcpe.bestMove16
    # see: move.hpp:30
    # xxxxxxxx xxxxxxxx xxxxxxxx x1111111  移動先
    # xxxxxxxx xxxxxxxx xx111111 1xxxxxxx  移動元。駒打ちの際には、PieceType + SquareNum - 1
    # xxxxxxxx xxxxxxxx x1xxxxxx xxxxxxxx  1 なら成り
    to_square = bestMove16 & 0b1111111
    from_square = (bestMove16 >> 7) & 0b1111111
    drop_piece_type = None
    if from_square >= 81:
        drop_piece_type = PieceType[from_square - 81]
    promotion = bestMove16 & 0b100000000000000 > 0
    if drop_piece_type:
        bestMove = shogi.Move(None, shogi.SQUARES_L90[to_square], drop_piece_type=drop_piece_type)
    else:
        bestMove = shogi.Move(shogi.SQUARES_L90[from_square], shogi.SQUARES_L90[to_square], promotion)

    if hcpe.gameResult == 2:
        win = -1
    else:
        win = hcpe.gameResult
    if board.turn == shogi.WHITE:
        win *= -1
        
    return board, hcpe.eval, bestMove, win
