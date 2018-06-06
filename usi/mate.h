#pragma once

// 奇数手詰めチェック
// 手番側が王手でないこと
// // 詰ます手を返すバージョン
Move mateMoveInOddPlyReturnMove(Position& pos, const int depth);

// 奇数手詰めチェック
// 手番側が王手でないこと
bool mateMoveInOddPly(Position& pos, const int depth);

// 偶数手詰めチェック
// 手番側が王手されていること
bool mateMoveInEvenPly(Position& pos, const int depth);
