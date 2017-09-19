#pragma once

// 奇数手詰めチェック
// 手番側が王手でないこと
bool mateMoveInOddPly(Position& pos, int depth);

// 偶数手詰めチェック
// 手番側が王手されていること
bool mateMoveInEvenPly(Position& pos, int depth);

// 3手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn3Ply(Position& pos);

// 2手詰めチェック
// 手番側が王手されていること
bool mateMoveIn2Ply(Position& pos);
