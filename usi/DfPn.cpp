#include "position.hpp"
#include "move.hpp"
#include "generateMoves.hpp"
#include "DfPnHash.h"
#include "DfPn.h"

const int DFPN_CHILD_MAX = 256;

struct child_dfpn_node_t {
	Move move;  // 着手する座標
	int index;   // インデックス
	unsigned int pn;
	unsigned int dn;
};

struct dfpn_node_t {
	int child_num;                            // 子ノードの数
	child_dfpn_node_t child[DFPN_CHILD_MAX];  // 子ノードの情報
};

// DfPnのノード
dfpn_node_t *dfpn_node;

const int NOT_EXPANDED = -1;
const unsigned int INF = UINT_MAX;

void mid_or(Position& pos, int index, const int ply, unsigned int &th_p, unsigned int &th_d, unsigned int &pn, unsigned int &dn);
void mid_and(Position& pos, int index, const int ply, unsigned int &th_p, unsigned int &th_d, unsigned int &pn, unsigned int &dn);

//  DfPn探索の初期設定  //
void InitializeDfPnSearch(void)
{
	// ハッシュ初期化
	InitializeDfPnHash();

	// DfPnのノードのメモリを確保
	dfpn_node = (dfpn_node_t *)malloc(sizeof(dfpn_node_t) * dfpn_hash_size);

	if (dfpn_node == nullptr) {
		std::cerr << "Cannot allocate memory !!" << std::endl;
		std::cerr << "You must reduce tree size !!" << std::endl;
		exit(1);
	}
}

// ハッシュを割り当て
int NewDfPnHashNode(Position& pos, const int ply)
{
	int index = SearchEmptyDfPnHashIndex(pos.getKey(), pos.turn(), pos.gamePly() + ply);
	dfpn_node[index].child_num = NOT_EXPANDED;

	if (!CheckRemainingDfPnHashSize()) {
		std::cout << "hash full" << std::endl;
	}

	return index;
}

// ハッシュを引く
int LookUpDfPnHash(Position& pos, const int ply)
{
	unsigned int index = FindSameDfPnHashIndex(pos.getKey(), pos.turn());
	child_dfpn_node_t *dfpn_child;

	// 登録済みの場合、それを返す
	if (index != dfpn_hash_size) {
		return index;
	}

	// 未登録の場合
	return NewDfPnHashNode(pos, ply);
}

Move dfpn(Position& pos, const int ply)
{
	// rootノードのみ特別なOR節点の処理

	// 1手詰みチェック
	Move move = pos.mateMoveIn1Ply();
	if (move != Move::moveNone()) {
		return move;
	}

	int index = LookUpDfPnHash(pos, ply);
	unsigned int th_p = INF - 1;
	unsigned int th_d = INF - 1;

	// OR節点
	// ノードの証明数pnが閾値th_p以上になるか、反証数dnが閾値th_d以上になるまで自分の下を探索し続ける。
	// 証明数最小の子ノードn_c、二番目に証明数の小さい子ノードn_2を選ぶ。
	// n_cの閾値を以下の通り設定する。
	//   n_c.th_p = min(th_p, n_2.pn + 1)
	//   n_c.th_d = th_d + n_c.dn - Σn_child.dn

	child_dfpn_node_t *dfpn_child = dfpn_node[index].child;

	// 末端ノードの場合
	if (dfpn_node[index].child_num == NOT_EXPANDED) {
		// 候補手の展開
		int child_num = 0;
		// 王手の指し手生成
		for (MoveList<Check> ml(pos); !ml.end(); ++ml) {
			dfpn_child[child_num].move = ml.move();
			dfpn_child[child_num].index = NOT_EXPANDED;
			dfpn_child[child_num].pn = 1;
			dfpn_child[child_num].dn = 1;
			child_num++;
		}
		// 王手がない場合
		if (child_num == 0) {
			// 不詰み
			return Move::moveNone();
		}

		dfpn_node[index].child_num = child_num;
	}

	// 反復深化
	while (true) {
		// min(pn_child)
		unsigned int min_pn = dfpn_child[0].pn;
		unsigned int min_pn2 = dfpn_child[0].pn;
		int n_c = 0;
		for (int i = 1; i < dfpn_node[index].child_num; i++) {
			if (dfpn_child[i].pn < min_pn) {
				min_pn2 = min_pn;
				min_pn = dfpn_child[i].pn;
				n_c = i;
			}
		}
		// Σdn_child
		unsigned int sum_dn = 0;
		for (int i = 0; i < dfpn_node[index].child_num; i++) {
			sum_dn += dfpn_child[i].dn;
		}

		// 閾値以上なら探索終了
		if (min_pn >= th_p || sum_dn >= th_d) {
			if (min_pn == 0) {
				return dfpn_child[n_c].move;
			}
			return Move::moveNone();
		}

		// 閾値を割り当て、証明数が最小の子ノードn_cを探索する
		unsigned int child_th_p;
		if (min_pn == INF - 1)
			child_th_p = INF;
		else
			child_th_p = std::min(th_p, min_pn2 == INF ? INF : min_pn2 + 1);

		unsigned int child_th_d;
		if (dfpn_child[n_c].dn == INF - 1)
			child_th_d = INF;
		else if (th_d >= INF - 1)
			child_th_d = INF - 1;
		else
			child_th_d = th_d + dfpn_child[n_c].dn - sum_dn;

		StateInfo state;
		pos.doMove(dfpn_child[n_c].move, state);
		//std::cout << ply << " : " << dfpn_child[n_c].move.toUSI() << std::endl;

		if (dfpn_child[n_c].index == NOT_EXPANDED) {
			dfpn_child[n_c].index = LookUpDfPnHash(pos, ply + 1);
		}
		mid_and(pos, dfpn_child[n_c].index, ply + 1, child_th_p, child_th_d, dfpn_child[n_c].pn, dfpn_child[n_c].dn);
		if (dfpn_child[n_c].pn == 0) {
			// 詰みが見つかった時点で終了
			pos.undoMove(dfpn_child[n_c].move);
			return dfpn_child[n_c].move;
		}

		pos.undoMove(dfpn_child[n_c].move);
	}
}

void mid_or(Position& pos, int index, const int ply, unsigned int &th_p, unsigned int &th_d, unsigned int &pn, unsigned int &dn)
{
	// OR節点
	// ノードの証明数pnが閾値th_p以上になるか、反証数dnが閾値th_d以上になるまで自分の下を探索し続ける。
	// 証明数最小の子ノードn_c、二番目に証明数の小さい子ノードn_2を選ぶ。
	// n_cの閾値を以下の通り設定する。
	//   n_c.th_p = min(th_p, n_2.pn + 1)
	//   n_c.th_d = th_d + n_c.dn - Σn_child.dn

	// 閾値チェック
	if (pn >= th_p || dn >= th_d) {
		th_p = pn;
		th_d = dn;
		return;
	}

	child_dfpn_node_t *dfpn_child = dfpn_node[index].child;

	// 末端ノードの場合
	if (dfpn_node[index].child_num == NOT_EXPANDED) {
		// 詰みチェック
		if (pos.mateMoveIn1Ply() != Move::moveNone()) {
			pn = 0;
			dn = INF;
			return;
		}

		// 候補手の展開
		int child_num = 0;
		// 王手の指し手生成
		for (MoveList<Check> ml(pos); !ml.end(); ++ml) {
			dfpn_child[child_num].move = ml.move();
			dfpn_child[child_num].index = NOT_EXPANDED;
			dfpn_child[child_num].pn = 1;
			dfpn_child[child_num].dn = 1;
			child_num++;
		}
		// 王手がない場合
		if (child_num == 0) {
			// 不詰み
			pn = INF;
			dn = 0;
			return;
		}

		dfpn_node[index].child_num = child_num;
	}

	// サイクル回避
	pn = th_p;
	dn = th_d;

	// 反復深化
	while (true) {
		// min(pn_child)
		unsigned int min_pn = dfpn_child[0].pn;
		unsigned int min_pn2 = dfpn_child[0].pn;
		int n_c = 0;
		for (int i = 1; i < dfpn_node[index].child_num; i++) {
			if (dfpn_child[i].pn < min_pn) {
				min_pn2 = min_pn;
				min_pn = dfpn_child[i].pn;
				n_c = i;
			}
		}
		// Σdn_child
		unsigned int sum_dn = 0;
		for (int i = 0; i < dfpn_node[index].child_num; i++) {
			sum_dn += dfpn_child[i].dn;
		}

		// 閾値以上なら探索終了
		if (min_pn >= th_p || sum_dn >= th_d) {
			pn = min_pn;
			dn = sum_dn;
			return;
		}

		// 閾値を割り当て、証明数が最小の子ノードn_cを探索する
		unsigned int child_th_p;
		if (min_pn == INF - 1)
			child_th_p = INF;
		else
			child_th_p = std::min(th_p, min_pn2 == INF ? INF : min_pn2 + 1);

		unsigned int child_th_d;
		if (dfpn_child[n_c].dn == INF - 1)
			child_th_d = INF;
		else if (th_d >= INF - 1)
			child_th_d = INF - 1;
		else
			child_th_d = th_d + dfpn_child[n_c].dn - sum_dn;

		StateInfo state;
		pos.doMove(dfpn_child[n_c].move, state);
		//std::cout << ply << " : " << dfpn_child[n_c].move.toUSI() << std::endl;

		if (dfpn_child[n_c].index == NOT_EXPANDED) {
			dfpn_child[n_c].index = LookUpDfPnHash(pos, ply + 1);
		}
		mid_and(pos, dfpn_child[n_c].index, ply + 1, child_th_p, child_th_d, dfpn_child[n_c].pn, dfpn_child[n_c].dn);
		if (dfpn_child[n_c].pn == 0) {
			// 詰みが見つかった時点で終了
			pn = 0;
			dn = INF;
			pos.undoMove(dfpn_child[n_c].move);
			return;
		}

		pos.undoMove(dfpn_child[n_c].move);
	}
}

void mid_and(Position& pos, int index, const int ply, unsigned int &th_p, unsigned int &th_d, unsigned int &pn, unsigned int &dn)
{
	// AND節点
	// ノードの証明数pnが閾値th_p以上になるか、反証数dnが閾値th_d以上になるまで自分の下を探索し続ける。
	// 反証数最小の子ノードn_c、二番目に反証数の小さい子ノードn_2を選ぶ。
	// n_cの閾値を以下の通り設定する。
	//   n_c.th_p = th_p + n_c.pn - Σn_child.pn
	//   n_c.th_d = min(th_d, n_2.dn + 1)

	// 閾値チェック
	if (pn >= th_p || dn >= th_d) {
		th_p = pn;
		th_d = dn;
		return;
	}

	child_dfpn_node_t *dfpn_child = dfpn_node[index].child;

	// 末端ノードの場合
	if (dfpn_node[index].child_num == NOT_EXPANDED) {
		// 候補手の展開
		int child_num = 0;
		// Evasionの指し手生成
		// OR節点で1手詰めチェックを行っているので必ず指し手がある
		for (MoveList<Legal> ml(pos); !ml.end(); ++ml) {
			dfpn_child[child_num].move = ml.move();
			dfpn_child[child_num].index = NOT_EXPANDED;
			dfpn_child[child_num].pn = 1;
			dfpn_child[child_num].dn = 1;
			child_num++;
		}

		// 不詰みチェック
		for (int i = 0; i < child_num; i++) {
			StateInfo state;
			pos.doMove(dfpn_child[i].move, state);
			if (pos.inCheck()) {
				// 自玉が王手の場合、不詰み
				pn = INF;
				dn = 0;
				pos.undoMove(dfpn_child[i].move);
				return;
			}
			pos.undoMove(dfpn_child[i].move);
		}

		dfpn_node[index].child_num = child_num;
	}

	// サイクル回避
	pn = th_p;
	dn = th_d;

	// 反復深化
	while (true) {
		// Σpn_child
		unsigned int sum_pn = 0;
		for (int i = 0; i < dfpn_node[index].child_num; i++) {
			sum_pn += dfpn_child[i].pn;
		}
		// min(dn_child)
		unsigned int min_dn = dfpn_child[0].dn;
		unsigned int min_dn2 = INF;
		int n_c = 0;
		for (int i = 1; i < dfpn_node[index].child_num; i++) {
			if (dfpn_child[i].dn < min_dn) {
				min_dn2 = min_dn;
				min_dn = dfpn_child[i].dn;
				n_c = i;
			}
		}

		// 閾値以上なら探索終了
		if (sum_pn >= th_p || min_dn >= th_d) {
			pn = sum_pn;
			dn = min_dn;
			return;
		}

		// 閾値を割り当て、証明数が最小の子ノードn_cを探索する
		unsigned int child_th_p;
		if (dfpn_child[n_c].pn == INF - 1)
			child_th_p = INF;
		else if (th_p >= INF - 1)
			child_th_p = INF - 1;
		else
			child_th_p = th_p + dfpn_child[n_c].pn - sum_pn;

		unsigned int child_th_d;
		if (min_dn == INF - 1)
			child_th_d = INF;
		else
			child_th_d = std::min(th_d, min_dn2 == INF ? INF : min_dn2 + 1);

		StateInfo state;
		pos.doMove(dfpn_child[n_c].move, state);
		//std::cout << ply << " : " << dfpn_child[n_c].move.toUSI() << std::endl;

		if (dfpn_child[n_c].index == NOT_EXPANDED) {
			dfpn_child[n_c].index = LookUpDfPnHash(pos, ply + 1);
		}
		mid_or(pos, dfpn_child[n_c].index, ply + 1, child_th_p, child_th_d, dfpn_child[n_c].pn, dfpn_child[n_c].dn);
		if (dfpn_child[n_c].dn == 0) {
			// 不詰みが見つかった時点で終了
			pn = INF;
			dn = 0;
			pos.undoMove(dfpn_child[n_c].move);
			return;
		}

		pos.undoMove(dfpn_child[n_c].move);
	}
}
