#pragma once

// 7手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn7Ply(Position& pos);

// 6手詰めチェック
// 手番側が王手されていること
bool mateMoveIn6Ply(Position& pos);

// 5手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn5Ply(Position& pos);

// 4手詰めチェック
// 手番側が王手されていること
bool mateMoveIn4Ply(Position& pos);

// 3手詰めチェック
// 手番側が王手でないこと
bool mateMoveIn3Ply(Position& pos);

// 2手詰めチェック
// 手番側が王手されていること
bool mateMoveIn2Ply(Position& pos);
